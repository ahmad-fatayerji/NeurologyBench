#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "net.h"

#include <string.h>
#include <stdint.h>
#include <float.h>
#include <errno.h>
#include <sys/stat.h>

static double now_seconds(void) {
#if defined(CLOCK_MONOTONIC)
	struct timespec ts;
	if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
		return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
	}
#endif
	return (double)clock() / (double)CLOCKS_PER_SEC;
}

static void usage(const char *exe) {
	printf("Usage: %s [--model cnn|fc] [--samples N] [--warmup N] [--runs N] [--threads N[,N...]] [--mnist-dir PATH] [--csv FILE]\n", exe);
	printf("Defaults: --model cnn --samples 100 --warmup 10 --runs 20 --threads 4 --mnist-dir MNIST\n");
}

static Network *build_cnn(void) {
	Network *cnn = initNetwork(cross_entropy_loss, cross_entropy_prime);
	if (!cnn) {
		return NULL;
	}

	Layer *conv1 = initConv2D(8, 3, 3, 1, 1, 0);
	Layer *act1 = initActivation(relu_activation, relu_p, 8 * 26 * 26);
	Layer *pool1 = initMaxPool(8, 26, 26, 2, 2, 2);
	Layer *fc1 = initFC(8 * 13 * 13, 128);
	Layer *act_fc1 = initActivation(relu_activation, relu_p, 128);
	Layer *fc2 = initFC(128, 10);

	if (!conv1 || !act1 || !pool1 || !fc1 || !act_fc1 || !fc2) {
		destroyNetwork(cnn);
		return NULL;
	}

	addLayer(cnn, conv1);
	addLayer(cnn, act1);
	addLayer(cnn, pool1);
	addLayer(cnn, fc1);
	addLayer(cnn, act_fc1);
	addLayer(cnn, fc2);
	return cnn;
}

static Network *build_fc(void) {
	Network *net = initNetwork(cross_entropy_loss, cross_entropy_prime);
	if (!net) {
		return NULL;
	}

	Layer *fc1 = initFC(28 * 28, 64);
	Layer *act1 = initActivation(relu_activation, relu_p, 64);
	Layer *fc2 = initFC(64, 64);
	Layer *act2 = initActivation(relu_activation, relu_p, 64);
	Layer *fc3 = initFC(64, 10);

	if (!fc1 || !act1 || !fc2 || !act2 || !fc3) {
		destroyNetwork(net);
		return NULL;
	}

	addLayer(net, fc1);
	addLayer(net, act1);
	addLayer(net, fc2);
	addLayer(net, act2);
	addLayer(net, fc3);
	return net;
}

typedef struct {
	double total_infers;
	double total_time_s;
	double avg_latency_ms;
	double min_latency_ms;
	double max_latency_ms;
	double throughput_inf_s;
} BenchmarkStats;

static int parse_thread_list(const char *arg, int *threads_out, int max_threads, int *count_out) {
	char buf[256];
	char *saveptr = NULL;
	int count = 0;

	if (!arg || !threads_out || !count_out || max_threads <= 0) {
		return 0;
	}

	snprintf(buf, sizeof(buf), "%s", arg);
	for (char *tok = strtok_r(buf, ",", &saveptr); tok; tok = strtok_r(NULL, ",", &saveptr)) {
		char *end = NULL;
		errno = 0;
		long v = strtol(tok, &end, 10);
		if (end == tok || *end != '\0' || errno == ERANGE || v <= 0 || v > INT_MAX) {
			return 0;
		}
		if (count >= max_threads) {
			return 0;
		}
		threads_out[count++] = (int)v;
	}

	if (count == 0) {
		return 0;
	}

	*count_out = count;
	return 1;
}

static BenchmarkStats run_benchmark(
	Network *net,
	int threads,
	const char *model,
	double *x_test_flat,
	int samples,
	int warmup,
	int runs,
	int channels,
	int height,
	int width
) {
	BenchmarkStats stats = {0};
	setThreadPoolSize(net, threads);

	printf("Benchmarking %s inference on %d samples (warmup=%d, runs=%d, threads=%d)\n",
		model, samples, warmup, runs, threads);

	for (int w = 0; w < warmup; ++w) {
		int idx = w % samples;
		double *out = infer_sample(net, x_test_flat + (size_t)idx * 28 * 28, channels, height, width);
		if (out) {
			free(out);
		}
	}

	double start = now_seconds();
	double min_latency_ms = DBL_MAX;
	double max_latency_ms = 0.0;
	for (int r = 0; r < runs; ++r) {
		for (int i = 0; i < samples; ++i) {
			double infer_start = now_seconds();
			double *out = infer_sample(net, x_test_flat + (size_t)i * 28 * 28, channels, height, width);
			double infer_end = now_seconds();
			double infer_ms = (infer_end - infer_start) * 1000.0;
			if (infer_ms < min_latency_ms) {
				min_latency_ms = infer_ms;
			}
			if (infer_ms > max_latency_ms) {
				max_latency_ms = infer_ms;
			}
			if (out) {
				free(out);
			}
		}
	}
	double end = now_seconds();

	stats.total_infers = (double)runs * (double)samples;
	stats.total_time_s = end - start;
	stats.avg_latency_ms = (stats.total_time_s / stats.total_infers) * 1000.0;
	stats.min_latency_ms = min_latency_ms;
	stats.max_latency_ms = max_latency_ms;
	stats.throughput_inf_s = stats.total_infers / stats.total_time_s;

	printf("Total inferences: %.0f\n", stats.total_infers);
	printf("Total time: %.6f s\n", stats.total_time_s);
	printf("Avg latency: %.3f ms\n", stats.avg_latency_ms);
	printf("Min latency: %.3f ms\n", stats.min_latency_ms);
	printf("Max latency: %.3f ms\n", stats.max_latency_ms);
	printf("Throughput: %.2f inf/s\n", stats.throughput_inf_s);

	return stats;
}

static int file_exists_and_nonempty(const char *path) {
	struct stat st;
	if (!path) {
		return 0;
	}
	if (stat(path, &st) != 0) {
		return 0;
	}
	return st.st_size > 0;
}

static void get_uname_a(char *out, size_t out_size) {
	FILE *fp = NULL;
	if (!out || out_size == 0) {
		return;
	}

	out[0] = '\0';
	fp = popen("uname -a", "r");
	if (!fp) {
		snprintf(out, out_size, "uname -a unavailable");
		return;
	}

	if (!fgets(out, (int)out_size, fp)) {
		snprintf(out, out_size, "uname -a unavailable");
	}
	pclose(fp);

	size_t len = strlen(out);
	while (len > 0 && (out[len - 1] == '\n' || out[len - 1] == '\r')) {
		out[--len] = '\0';
	}
}

int main(int argc, char **argv) {
	const char *model = "cnn";
	const char *mnist_dir = "MNIST";
	const char *csv_path = NULL;
	int samples = 100;
	int warmup = 10;
	int runs = 20;
	int thread_values[16] = {4};
	int thread_count = 1;

	for (int i = 1; i < argc; ++i) {
		if (strcmp(argv[i], "--help") == 0) {
			usage(argv[0]);
			return 0;
		}
		if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
			model = argv[++i];
			continue;
		}
		if (strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
			samples = atoi(argv[++i]);
			continue;
		}
		if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
			warmup = atoi(argv[++i]);
			continue;
		}
		if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
			runs = atoi(argv[++i]);
			continue;
		}
		if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
			if (!parse_thread_list(argv[++i], thread_values, 16, &thread_count)) {
				printf("ERROR: Invalid --threads value. Use positive integers, e.g. 1 or 1,4\n");
				return 1;
			}
			continue;
		}
		if (strcmp(argv[i], "--mnist-dir") == 0 && i + 1 < argc) {
			mnist_dir = argv[++i];
			continue;
		}
		if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
			csv_path = argv[++i];
			continue;
		}
		printf("Unknown option: %s\n", argv[i]);
		usage(argv[0]);
		return 1;
	}

	if (samples <= 0 || warmup < 0 || runs <= 0 || thread_count <= 0) {
		printf("ERROR: Invalid numeric arguments.\n");
		return 1;
	}

	srand(0);

	char img_path[512];
	char uname_info[512];
	snprintf(img_path, sizeof(img_path), "%s/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte", mnist_dir);
	get_uname_a(uname_info, sizeof(uname_info));
	printf("System: %s\n", uname_info);

	int number_of_images = 0;
	double (*testImages)[28][28] = load_mnist_images(img_path, &number_of_images);
	if (!testImages || number_of_images <= 0) {
		printf("ERROR: Failed to load MNIST images from %s\n", img_path);
		return 1;
	}

	if (samples > number_of_images) {
		samples = number_of_images;
	}

	double *x_test_flat = malloc((size_t)samples * 28 * 28 * sizeof(double));
	if (!x_test_flat) {
		free(testImages);
		return 1;
	}
	for (int i = 0; i < samples; ++i) {
		for (int r = 0; r < 28; ++r) {
			for (int c = 0; c < 28; ++c) {
				x_test_flat[i * 28 * 28 + r * 28 + c] = testImages[i][r][c] / 255.0;
			}
		}
	}

	Network *net = NULL;
	int channels = 1;
	int height = 28;
	int width = 28;
	if (strcmp(model, "fc") == 0) {
		net = build_fc();
		channels = 1;
		height = 1;
		width = 28 * 28;
	} else if (strcmp(model, "cnn") == 0) {
		net = build_cnn();
	} else {
		printf("ERROR: Unknown model '%s'\n", model);
		free(x_test_flat);
		free(testImages);
		return 1;
	}

	if (!net) {
		printf("ERROR: Failed to build model\n");
		free(x_test_flat);
		free(testImages);
		return 1;
	}

	if (csv_path) {
		int has_rows = file_exists_and_nonempty(csv_path);
		FILE *csv = fopen(csv_path, "a");
		if (!csv) {
			printf("ERROR: Failed to open CSV file: %s\n", csv_path);
		} else {
			if (!has_rows) {
				fprintf(csv, "model,samples,warmup,runs,threads,total_infers,total_time_s,avg_latency_ms,min_latency_ms,max_latency_ms,throughput_inf_s,uname_a\n");
			}
			for (int t = 0; t < thread_count; ++t) {
				BenchmarkStats stats = run_benchmark(
					net, thread_values[t], model, x_test_flat, samples, warmup, runs, channels, height, width);
				fprintf(csv, "%s,%d,%d,%d,%d,%.0f,%.6f,%.3f,%.3f,%.3f,%.2f,\"%s\"\n",
					model, samples, warmup, runs, thread_values[t], stats.total_infers, stats.total_time_s, stats.avg_latency_ms,
					stats.min_latency_ms, stats.max_latency_ms, stats.throughput_inf_s, uname_info);
			}
			fclose(csv);
			printf("CSV appended to: %s\n", csv_path);
		}
	} else {
		for (int t = 0; t < thread_count; ++t) {
			run_benchmark(net, thread_values[t], model, x_test_flat, samples, warmup, runs, channels, height, width);
		}
	}

	destroyNetwork(net);
	free(x_test_flat);
	free(testImages);
	return 0;
}

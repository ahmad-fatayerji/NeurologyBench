#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "net.h"

#include <string.h>
#include <stdint.h>
#include <float.h>

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
	printf("Usage: %s [--model cnn|fc] [--samples N] [--warmup N] [--runs N] [--threads N] [--mnist-dir PATH] [--csv FILE]\n", exe);
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

int main(int argc, char **argv) {
	const char *model = "cnn";
	const char *mnist_dir = "MNIST";
	const char *csv_path = NULL;
	int samples = 100;
	int warmup = 10;
	int runs = 20;
	int threads = 4;

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
			threads = atoi(argv[++i]);
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

	if (samples <= 0 || warmup < 0 || runs <= 0 || threads <= 0) {
		printf("ERROR: Invalid numeric arguments.\n");
		return 1;
	}

	srand(0);

	char img_path[512];
	snprintf(img_path, sizeof(img_path), "%s/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte", mnist_dir);

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

	double total_infers = (double)runs * (double)samples;
	double total_time = end - start;
	double per_infer_ms = (total_time / total_infers) * 1000.0;
	double throughput = total_infers / total_time;

	printf("Total inferences: %.0f\n", total_infers);
	printf("Total time: %.6f s\n", total_time);
	printf("Avg latency: %.3f ms\n", per_infer_ms);
	printf("Min latency: %.3f ms\n", min_latency_ms);
	printf("Max latency: %.3f ms\n", max_latency_ms);
	printf("Throughput: %.2f inf/s\n", throughput);

	if (csv_path) {
		FILE *csv = fopen(csv_path, "w");
		if (!csv) {
			printf("ERROR: Failed to open CSV file: %s\n", csv_path);
		} else {
			fprintf(csv, "model,samples,warmup,runs,threads,total_infers,total_time_s,avg_latency_ms,min_latency_ms,max_latency_ms,throughput_inf_s\n");
			fprintf(csv, "%s,%d,%d,%d,%d,%.0f,%.6f,%.3f,%.3f,%.3f,%.2f\n",
				model, samples, warmup, runs, threads, total_infers, total_time, per_infer_ms, min_latency_ms, max_latency_ms, throughput);
			fclose(csv);
			printf("CSV written to: %s\n", csv_path);
		}
	}

	destroyNetwork(net);
	free(x_test_flat);
	free(testImages);
	return 0;
}

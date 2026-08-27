#ifndef MLC_DATASET_H
#define MLC_DATASET_H
#include <mlc/matrix.h>
#include <stdbool.h>

typedef struct {
    MLCMatrix features;
    MLCMatrix targets;
} MLCDataset;

MLCStatus mlc_dataset_create(size_t sample_count, size_t feature_count, size_t targe_count, MLCDataset *output_dataset);
void mlc_dataset_free(MLCDataset *dataset);
MLCStatus mlc_dataset_load_csv(const char *filepath, size_t target_count, bool has_header, MLCDataset *output_dataset);


#endif //MLC_DATASET_H
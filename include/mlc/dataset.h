#ifndef MLC_DATASET_H
#define MLC_DATASET_H
#include <mlc/matrix.h>
#include <stdbool.h>

typedef struct {
    MLCMatrix features;
    MLCMatrix targets;
} MLCDataset;

typedef struct {
    MLCMatrix means;
    MLCMatrix standard_deviations;
} MLCStandardScaler;

MLCStatus mlc_dataset_create(size_t sample_count, size_t feature_count, size_t targe_count, MLCDataset *output_dataset);
void mlc_dataset_free(MLCDataset *dataset);
MLCStatus mlc_dataset_load_csv(const char *filepath, size_t target_count, bool has_header, MLCDataset *output_dataset);
MLCStatus mlc_dataset_shuffle(MLCDataset *dataset, uint32_t seed);
MLCStatus mlc_dataset_train_test_split(const MLCDataset *dataset, double training_ratio, MLCDataset *training_dataset, MLCDataset *testing_dataset);
MLCStatus mlc_standard_scaler_fit(const MLCDataset *dataset, MLCStandardScaler *scaler);
void mlc_standard_scaler_free(MLCStandardScaler *scaler);
MLCStatus mlc_standard_scaler_transform(const MLCStandardScaler *scaler, MLCDataset *dataset);
#endif //MLC_DATASET_H
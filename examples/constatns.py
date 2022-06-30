import os

OUTPUT_DIR = 'output'
DATASETS_DIR = 'datasets'
VECTORS_DIR = 'vectors'
SUMMARY_DIR = 'summary'
EXPORT_DIR = 'export'
TABLES_DIR = 'tables'
FIGURES_DIR = 'figures'


def get_paths(ex_name):
    return os.path.join(ex_name, OUTPUT_DIR, DATASETS_DIR), \
           os.path.join(ex_name, OUTPUT_DIR, VECTORS_DIR), \
           os.path.join(ex_name, OUTPUT_DIR, SUMMARY_DIR), \
           os.path.join(ex_name, OUTPUT_DIR, SUMMARY_DIR, 'datasets_meta.csv'), \
           os.path.join(ex_name, OUTPUT_DIR, SUMMARY_DIR, 'setups_meta.csv'), \
           os.path.join(ex_name, OUTPUT_DIR, SUMMARY_DIR, 'embeddings_meta.csv'), \
           os.path.join(ex_name, OUTPUT_DIR, SUMMARY_DIR, 'predictions.csv'), \
           os.path.join(ex_name, OUTPUT_DIR, SUMMARY_DIR, 'metrics.csv'), \
           os.path.join(ex_name, OUTPUT_DIR, SUMMARY_DIR, 'aggregated.csv'), \
           os.path.join(ex_name, OUTPUT_DIR, EXPORT_DIR, TABLES_DIR), \
           os.path.join(ex_name, OUTPUT_DIR, EXPORT_DIR, FIGURES_DIR), \
           os.path.join(ex_name, OUTPUT_DIR, SUMMARY_DIR, 'hardware.json')


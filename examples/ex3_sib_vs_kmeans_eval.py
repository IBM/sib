import os.path
import random
import re
from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from sib.clustering_utils import reindex_labels
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, \
    v_measure_score, precision_recall_fscore_support, homogeneity_score, completeness_score
from tqdm import tqdm

from constatns import get_paths
from ex3_sib_vs_kmeans_setup import EMBEDDING_VIEW_ORDER, ALGORITHM_VIEW_ORDER, ALGORITHMS, EMBEDDINGS

tqdm.pandas()

random.seed(1024)

EX_NAME = 'ex3'

DATASETS_FULL_PATH, VECTORS_FULL_PATH, SUMMARY_FULL_PATH, \
    DATASETS_METADATA_FULL_PATH, SETUPS_METADATA_FULL_PATH, \
    EMBEDDINGS_FULL_PATH, PREDICTIONS_FULL_PATH, \
    METRICS_FULL_PATH, AGGREGATED_FULL_PATH, \
    TABLES_FULL_PATH, FIGURES_FULL_PATH, HARDWARE_PATH = get_paths(EX_NAME)


def classification_scores(row):
    gold_array = np.array(row['gold_labels'])
    predicted_array = np.array(row['predicted_labels'])

    row['ami'] = adjusted_mutual_info_score(gold_array, predicted_array)
    row['ari'] = adjusted_rand_score(gold_array, predicted_array)
    row['homogeneity'] = homogeneity_score(gold_array, predicted_array)
    row['completeness'] = completeness_score(gold_array, predicted_array)
    row['v-measure'] = v_measure_score(gold_array, predicted_array)

    reindexed_predicted_array = reindex_labels(gold_array, predicted_array, predicted_array)

    micro_pr, micro_rc, micro_f1, _ = \
        precision_recall_fscore_support(gold_array, reindexed_predicted_array, average='micro', zero_division=0)

    macro_pr, macro_rc, macro_f1, _ = \
        precision_recall_fscore_support(gold_array, reindexed_predicted_array, average='macro', zero_division=0)

    weighted_pr, weighted_rc, weighted_f1, _ = \
        precision_recall_fscore_support(gold_array, reindexed_predicted_array, average='weighted', zero_division=0)

    row['micro_pr'], row['macro_pr'], row['weighted_pr'] = micro_pr, macro_pr, weighted_pr
    row['micro_rc'], row['macro_rc'], row['weighted_rc'] = micro_rc, macro_rc, weighted_rc
    row['micro_f1'], row['macro_f1'], row['weighted_f1'] = micro_f1, macro_f1, weighted_f1

    return row


def compute_metrics():
    if not os.path.exists(METRICS_FULL_PATH):
        df = pd.read_csv(PREDICTIONS_FULL_PATH)
        print("Processing %d predictions" % df.shape[0])

        print("Parsing string lists of gold labels")
        df['gold_labels'] = df['gold_labels'].progress_apply(literal_eval)

        print("Parsing string lists of predicted labels")
        df['predicted_labels'] = df['predicted_labels'].progress_apply(literal_eval)

        print("Evaluating predictions")
        df = df.progress_apply(lambda row: classification_scores(row), axis=1)

        print("Dropping unneeded columns")
        df = df.drop(labels=['gold_labels', 'predicted_labels'], axis=1)

        df.to_csv(METRICS_FULL_PATH)


def bootstrap_ci95(series):
    mean = series.mean()
    if np.allclose(series, series.mean()):
        ci = mean, mean
    else:
        ci = bootstrap((series,), np.mean,
                       random_state=random.randint(0, 1024 ** 2)).confidence_interval
    return mean, ci[0], ci[1]


def split_ci_values(row):
    for measure, (mean, low, high) in row.items():
        row[measure] = mean
        row[measure + '_low'] = low
        row[measure + '_high'] = high
    return row


def aggregate_runs():
    if not os.path.exists(AGGREGATED_FULL_PATH):
        df = pd.read_csv(METRICS_FULL_PATH)
        measures = df.columns[df.dtypes == np.float64]
        df = df.groupby(['dataset', 'algorithm', 'algorithm_params',
                         'vectorizer_name', 'vectorizer_params']).agg(
            {measure: bootstrap_ci95 for measure in measures})
        df = df[measures].apply(split_ci_values, axis=1)
        df.to_csv(AGGREGATED_FULL_PATH)


def extract_params(param, row):
    for key, value in literal_eval(row[param]).items():
        row[key] = value
    return row


params_desc = {
    'max_iter': lambda x: 'Max iterations: %d' % x,
    'max_features': lambda x: 'Vocab size: %dk' % (x / 1000),
    'stop_words': lambda x: 'Stop words: ' +
                            ('filtered' if x and x == 'english' else 'allowed')
}


def get_setup_display_name(setup_params):
    desc = []
    for params in setup_params:
        for key, value in literal_eval(params).items():
            if key in params_desc:
                desc.append(params_desc[key](value))
            else:
                raise ValueError("Unknown parameter: %s" % key)

    return ', '.join(desc)


dataset_newline_re = re.compile(r'(.+) ([^ ]+)')


def create_plots():
    df = pd.read_csv(AGGREGATED_FULL_PATH)

    # sort by the parameters for better arrangement in charts
    df = df.apply(lambda row: extract_params('algorithm_params', row), axis=1)
    df = df.apply(lambda row: extract_params('vectorizer_params', row), axis=1)

    measures = {'ami': ('Adjusted Mutual Information', ''),
                'ari': ('Adjusted Rand-Index', ''),
                'v-measure': ('V-Measure', ''),
                'micro_f1': ('Micro Average F1', ''),
                'macro_f1': ('Macro Average F1', ''),
                'total_time': ('Total Run-Time', '')}

    groups = [[['ami', 'ari']],
              [['micro_f1', 'macro_f1'],
               ['v-measure', 'total_time']]]
    group_figsize = [(17, 8), (17, 16)]
    group_padding = [(None, 0.03), (0.04, 0.03)]
    time_measures = {'total_time'}

    # Okabe and Ito 2008 palette
    colors = ['#000000', '#E69F00', '#56B4E9', '#009E73',
              '#F0E442', '#0072B2', '#D55E00', '#CC79A7']

    # keep the black for the error bars
    colors = colors[1:]

    # figsize = (12, 10)
    bar_width = 0.22
    margin = 0.01

    df = sort_df(df)

    datasets = [dataset for dataset in df['dataset'].unique()]

    setup_groups = df.groupby(['algorithm', 'vectorizer_name',
                               'algorithm_params', 'vectorizer_params'], sort=False)

    ind_labels = np.arange(len(datasets)) * (1 + bar_width)  # the x locations for the groups
    ind_values = ind_labels - bar_width * (len(setup_groups)/2 - 0.5)

    labels = [f'{algorithm} on top of {embedding}' for (algorithm, embedding, c, d), _ in setup_groups]

    for group, figsize, padding in zip(groups, group_figsize, group_padding):
        rows = len(group)
        cols = len(group[0])
        hspace, wspace = padding
        fig, axes = plt.subplots(rows, cols, figsize=figsize,
                                 constrained_layout=True)
        fig.set_constrained_layout_pads(hspace=hspace, wspace=wspace)
        axes = axes.flatten()
        group = [y for x in group for y in x]
        for index, (measure, ax) in enumerate(zip(group, axes)):
            title, caption = measures[measure]

            log_scale = measure in time_measures
            upper_bound = None if log_scale else 1

            ax.margins(x=margin)

            if log_scale:
                ax.set_yscale('log', base=2)
                ax.set_yticks([])

            for setup_id, (_, df_setup) in enumerate(setup_groups):
                df_setup = df_setup.sort_values(by=['stop_words']).reset_index(drop=True)
                mean_values = df_setup[measure]
                ci_values = df_setup[[measure + '_low', measure + '_high']].T
                error_bar = (ci_values - mean_values).abs()
                rects = ax.bar(ind_values + setup_id*bar_width,
                               mean_values, yerr=error_bar.to_numpy(),
                               width=bar_width, capsize=3, ecolor='black',
                               color=colors[setup_id % len(colors)])

                if log_scale:
                    rotate = df_setup['vectorizer_name'].iloc[0] != EMBEDDINGS['sbert'][0]
                    autolabel(ax, rects, rotate)

            ax.set_title(title, fontsize=14)
            if upper_bound:
                ax.set_ylim(0.0, upper_bound)
                ax.set_yticks(np.arange(0, upper_bound + 0.1, 0.1))
            ax.set_xticks(ind_labels)
            ax.set_xticklabels(datasets, fontsize=14)

            ax.set_axisbelow(True)
            ax.grid(alpha=0.75, axis='y')

        fig.legend(labels=labels, loc='center', ncol=5,
                   bbox_to_anchor=(-0.05, -0.08), prop={'size': 14},
                   bbox_transform=axes[-1].transAxes)
        # fig.legend(labels=labels, loc='center', ncol=3,
        #            bbox_to_anchor=(0.5, -0.2), prop={'size': 14},
        #            bbox_transform=ax.transAxes)

        fig.savefig(os.path.join(FIGURES_FULL_PATH, f'fig_{"_".join(group)}.pdf'),
                    bbox_inches='tight', dpi=300)


def sort_df(df):
    orders = {
        'algorithm': {ALGORITHMS[a][0]: i for i, a in enumerate(ALGORITHM_VIEW_ORDER)},
        'vectorizer_name': {EMBEDDINGS[e][0]: i for i, e in enumerate(EMBEDDING_VIEW_ORDER)}
    }
    df = df.sort_values(by=['algorithm', 'vectorizer_name'],
                        key=lambda x: x.map(orders[x.name]))
    return df


def format_seconds(seconds):
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    if m > 60:
        h, m = divmod(m, 60)
        return f'{h:02d}:{m:02d}:{round(s):02d}'
    else:
        return f'{m:02d}:{round(s):02d}'


def autolabel(ax, rects, rotate):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        value = format_seconds(height)
        rotation = 45 if rotate else 0
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height, value,
                # '%.2f' % height,
                ha='center', va='bottom', fontsize=9, rotation=rotation)


def create_results_table():
    # sets = [{'name': 'table_results.tex',
    #          'caption': 'Assessment of clustering quality using the '
    #                     'AMI, ARI, V-Measure (\\textit{VM}), Micro-F1 (\\textit{Mic-F1}) '
    #                     'and Macro-F1 (\\textit{Mac-F1}) metrics',
    #          'label': 'tab:results',
    #          'metrics': {'ami': 'AMI', 'ari': 'ARI', 'v-measure': 'VM',
    #                      'micro_f1': 'Mic-F1', 'macro_f1': 'Mac-F1'},
    #          'columns': {'dataset': 'Dataset', 'algorithm': 'Algorithm',
    #                      'vectorizer_name': 'Embed.'},
    #          'min_max_func': pd.Series.max,
    #          'wide': False},
    #         {'name': 'table_times.tex',
    #          'caption': 'Assessment of clustering speed based on measurements of the '
    #                     'vectorization time (\\textit{Vector.}) and clustering time (\\textit{Cluster.}), '
    #                     'and their sum (\\textit{Total})',
    #          'label': 'tab:times',
    #          'metrics': {'vectorizer_time': 'Vector.', 'algorithm_time': 'Cluster.', 'total_time': 'Total'},
    #          'columns': {'dataset': 'Dataset', 'algorithm': 'Algorithm', 'vectorizer_name': 'Embed.'},
    #          'min_max_func': pd.Series.min,
    #          'wide': False},
    sets = [{'name': 'table_results.tex',
             'caption': 'Assessment of clustering quality using the metrics: '
                        'AMI, ARI, V-Measure (\\textit{VM}), Micro-F1 (\\textit{Mic-F1}) '
                        'and Macro-F1 (\\textit{Mac-F1}), and of clustering speed based '
                        'on measurements of the vectorization time (\\textit{Vector.}), '
                        'clustering time (\\textit{Cluster.}), and their sum (\\textit{Total})',
             'label': 'tab:results',
             'metrics': {'ami': 'AMI', 'ari': 'ARI', 'v-measure': 'VM',
                         'micro_f1': 'Mic-F1', 'macro_f1': 'Mac-F1',
                         'vectorizer_time': 'Vector.', 'algorithm_time': 'Cluster.',
                         'total_time': 'Total'},
             'columns': {'dataset': 'Dataset', 'algorithm': 'Algorithm',
                         'vectorizer_name': 'Embed.'},
             'min_max_func': pd.Series.max,
             'wide': True}
            ]

    time_metrics = {'vectorizer_time', 'algorithm_time', 'total_time'}

    df = pd.read_csv(AGGREGATED_FULL_PATH)
    df = sort_df(df)

    for table in sets:
        metrics = table['metrics']
        columns = table['columns']
        caption = table['caption']
        label = table['label']
        wide = table['wide']

        columns.update(metrics)

        opening, fields_spec, ending, columns_names = prepare_table(columns, metrics, caption, label, wide)

        body = ''
        for dataset, df_group in df.groupby('dataset'):
            body += f"\\multirow[t]{{{len(df_group)}}}{{*}}{dataset}\n"
            bolds = {}
            for metric in metrics.keys():
                values = df_group[metric].round(0 if metric in time_metrics else 2)
                func = pd.Series.min if metric in time_metrics else pd.Series.max
                bolds[metric] = pd.Series(np.isclose(values, func(values)), values.index)
            for i, row in df_group.iterrows():
                body += f" & {row['algorithm']} & {row['vectorizer_name']} & "
                metrics_latex = []
                for metric in metrics:
                    metric_latex = f'{format_seconds(row[metric])}' if metric in time_metrics else f'{row[metric]:.2f}'
                    if bolds[metric][i]:
                        metric_latex = f"\\textbf{{" + metric_latex + f"}}"
                    metrics_latex.append(metric_latex)
                body += " & ".join(metrics_latex)
                body += " \\\\\n"
            body += "\\hline\n"
        latex = f'{opening}{fields_spec}\n{columns_names}\n{body}\n{ending}'
        with open(os.path.join(TABLES_FULL_PATH, table['name']), "wt") as f:
            f.write(latex)


def create_datasets_table():

    fields = {
        # 'n_samples_org': '\\# Texts',
        # 'n_samples': '\\# Used',
        'n_samples': '\\# Texts',
        'word_count_mean': '\\# Words',
        'n_clusters': '\\# Classes',
    }
    columns = {
        'dataset': 'Dataset'
    }
    columns.update(fields)

    df = pd.read_csv(DATASETS_METADATA_FULL_PATH)
    df = df.sort_values(by='n_samples_org')

    # 'The column \\textit{\\# Used} indicates the number of texts used in this evaluation. ' \
    caption = 'Benchmark datasets for evaluation. ' \
              'The column \\textit{\\# Texts} indicates the number of texts in the dataset. ' \
              'The column \\textit{\\# Words} shows the average text length in terms of word-count in the dataset, ' \
              'and \\textit{\\# Classes} shows the number of classes in the dataset.'
    label = 'tab:datasets'

    opening, fields_spec, ending, columns_names = prepare_table(columns, fields, caption, label, False)

    body = ''
    for i, row in df.iterrows():
        values = [int(round(row[col])) if isinstance(row[col], float) else row[col] for col in columns]
        values = [f'{v:,}' if isinstance(v, int) else v for v in values]
        body += " & ".join(values)
        body += " \\\\\n"
        body += "\\hline\n"

    latex = f'{opening}{fields_spec}\n{columns_names}\n{body}\n{ending}'
    with open(os.path.join(TABLES_FULL_PATH, 'table_datasets.tex'), "wt") as f:
        f.write(latex)


def prepare_table(columns, fields, caption, label, wide):
    wide_opening = f'\\begin{{adjustwidth}}{{-\\extralength}}{{0cm}}\n' if wide else ''
    wide_ending = f'\\end{{adjustwidth}}\n' if wide else ''
    opening = f"\\begin{{table}}[]\n\\caption{{{caption}}}\n\\centering\n{wide_opening}\\begin{{tabular}}"
    ending = f"\\end{{tabular}}\n{wide_ending}\\label{{{label}}}\n\\end{{table}}"
    fields_spec = f"{{|" + " | ".join(["l"] * (len(columns) - len(fields)) + ["c"] * len(fields)) + f"|}}\n"
    columns_names = "\\hline\n" + " & ".join(map(lambda x: '\\textbf{' + x + '}', columns.values())) + "\\\\\\hline\n"
    return opening, fields_spec, ending, columns_names


def main():
    compute_metrics()
    aggregate_runs()
    create_plots()
    create_results_table()
    create_datasets_table()


if __name__ == '__main__':
    main()

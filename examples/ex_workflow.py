import gzip
import json
import os
import pickle
import platform
import random
import sys
from ast import literal_eval
from time import time

import numpy as np
import pandas as pd
import psutil
from matplotlib import pyplot as plt
from scipy.stats import bootstrap
from sib.clustering_utils import reindex_labels
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, homogeneity_score, completeness_score, \
    v_measure_score, precision_recall_fscore_support
from tqdm import tqdm

from constatns import get_paths
from vectorizers import GloveVectorizer

tqdm.pandas()


class ExampleWorkflow:

    def __init__(self, ex_name, datasets, embeddings, algorithms, setups, dataset_max_size,
                 n_runs, algorithm_view_order, embedding_view_order, seed):
        self.ex_name = ex_name
        self.datasets = datasets
        self.embeddings = embeddings
        self.algorithms = algorithms
        self.setups = setups
        self.dataset_max_size = dataset_max_size
        self.algorithm_view_order = algorithm_view_order
        self.embedding_view_order = embedding_view_order
        self.datasets_full_path, self.vectors_full_path, self.summary_full_path, \
            self.datasets_metadata_full_path, self.setups_metadata_full_path, \
            self.embeddings_meta_full_path, self.predictions_full_path, \
            self.metrics_full_path, self.aggregated_full_path, \
            self.tables_full_path, self.figures_full_path, self.hardware_path = get_paths(ex_name)
        self.n_runs = n_runs
        self.hidden_params = ['n_clusters', 'random_state']
        self.run_seeds = np.random.RandomState(seed).randint(np.iinfo(np.int32).max, size=n_runs)

    def prepare_dirs_model(self):
        os.makedirs(os.path.join(self.datasets_full_path), exist_ok=True)
        os.makedirs(os.path.join(self.vectors_full_path), exist_ok=True)
        os.makedirs(os.path.join(self.summary_full_path), exist_ok=True)
        os.makedirs(os.path.join(self.tables_full_path), exist_ok=True)
        os.makedirs(os.path.join(self.figures_full_path), exist_ok=True)
        GloveVectorizer.download_model()

    def record_setups(self):
        data = []

        # go over all setups
        for setup in self.setups:
            algorithm_name, embedding_type = setup
            algorithm_display_name = self.algorithms[algorithm_name][0]
            embedding_display_name, _, _ = self.embeddings[embedding_type]
            data.append({
                'algorithm': algorithm_display_name,
                'embedding': embedding_display_name
            })

        # create a dataframe
        df = pd.DataFrame(data)

        # append to previous runs if exist
        if os.path.exists(self.setups_metadata_full_path):
            df_prev = pd.read_csv(self.setups_metadata_full_path)
            df_prev.drop(df_prev.filter(regex="Unnamed"), axis=1, inplace=True)
            df = pd.concat([df_prev, df], ignore_index=True)

        # store the metadata of the setups
        df.to_csv(self.setups_metadata_full_path)

    def read_datasets(self):
        data = []

        # read all datasets and save into pickle
        for dataset_type, dataset_desc in self.datasets.items():
            dataset_name, dataset_fetcher = dataset_desc
            dataset_file_name = dataset_type + '.pkl.gz'
            dataset_full_path = os.path.join(self.datasets_full_path, dataset_file_name)
            if not os.path.exists(dataset_full_path):
                print("Reading dataset: %s..." % dataset_name, end=" ", flush=True)
                t0 = time()
                dataset = dataset_fetcher(self.dataset_max_size)
                print("done in %.3f sec." % (time() - t0), end=" ", flush=True)
                print("%d samples, %d classes" % (dataset.n_samples, dataset.n_clusters))
                with gzip.open(dataset_full_path, 'wb') as fp:
                    pickle.dump(dataset, fp)
                data.append({
                    'dataset': dataset_name,
                    'n_samples': dataset.n_samples,
                    'n_samples_org': dataset.n_samples_org,
                    'word_count_mean': dataset.word_count_mean,
                    'word_count_std': dataset.word_count_std,
                    'word_count_median': dataset.word_count_median,
                    'n_clusters': dataset.n_clusters,
                    'topics': dataset.target_names
                })

        # store the metadata of the datasets
        pd.DataFrame(data).to_csv(self.datasets_metadata_full_path)

    def vectorize(self):

        # vectorize all datasets for all setups
        if not os.path.exists(self.embeddings_meta_full_path):
            embedding_results = []

            for dataset_type, dataset_desc in self.datasets.items():
                dataset_name, dataset_fetcher = dataset_desc
                dataset_file_name = dataset_type + '.pkl.gz'
                dataset_full_path = os.path.join(self.datasets_full_path, dataset_file_name)
                print("Reading dataset: %s from pickle" % dataset_name)
                with gzip.open(dataset_full_path, 'rb') as fp:
                    dataset = pickle.load(fp)
                    texts = dataset.data

                for embedding_type, embedding_desc in self.embeddings.items():
                    vectorizer_name, vectorizer_factory, vectorizer_params = embedding_desc
                    vectorizer_kwargs_list = self.populate_kwargs({}, vectorizer_params)
                    print(f'\tEmbedding: {embedding_type}')
                    for vectorizer_kwargs in vectorizer_kwargs_list:
                        vectors_file_name = embedding_type + '_' + self.str_kwargs(vectorizer_kwargs) + '.pkl'
                        os.makedirs(os.path.join(self.vectors_full_path, dataset_type), exist_ok=True)
                        vectors_full_path = os.path.join(self.vectors_full_path, dataset_type, vectors_file_name)
                        print(f"\t\tWith: {str(vectorizer_kwargs) if len(vectorizer_kwargs)>0 else 'defaults'}...",
                              end=' ')
                        t0 = time()
                        vectorizer_instance = vectorizer_factory(**vectorizer_kwargs)
                        vectors = vectorizer_instance.fit_transform(texts)
                        vectorizer_time = time() - t0
                        embedding_results.append({
                            'dataset': dataset_name,
                            'embedding': embedding_type,
                            'vectorizer_name': vectorizer_name,
                            'vectorizer_params': vectorizer_kwargs,
                            'vectorizer_time': vectorizer_time,
                            'vectors_file_name': vectors_file_name,
                        })
                        print(f'done in: {vectorizer_time:.3f} secs.')
                        with open(vectors_full_path, 'wb') as fp:
                            pickle.dump(vectors, fp)

            # store the metadata of the embeddings
            pd.DataFrame(embedding_results).to_csv(self.embeddings_meta_full_path)

    def record_hardware(self):
        hardware_desc = self.get_system_desc()
        with open(self.hardware_path, 'wt') as f:
            json.dump(hardware_desc, f, indent=4)

    def cluster_setups(self):
        df_embeddings = pd.read_csv(self.embeddings_meta_full_path)

        results = []

        for dataset_type, dataset_desc in self.datasets.items():
            dataset_name, dataset_fetcher = dataset_desc
            dataset_file_name = dataset_type + '.pkl.gz'
            dataset_full_path = os.path.join(self.datasets_full_path, dataset_file_name)
            print("Reading dataset: %s from pickle" % dataset_name)
            with gzip.open(dataset_full_path, 'rb') as fp:
                dataset = pickle.load(fp)

            # run all setups
            for setup in self.setups:
                algorithm_name, embedding_type = setup

                if isinstance(algorithm_name, tuple):
                    algorithm_full_name = " + ".join([self.algorithms[name][0] for name in list(algorithm_name)])
                else:
                    algorithm_full_name = self.algorithms[algorithm_name][0]

                print("\tRunning %s" % algorithm_full_name, flush=True)

                embedding_desc = self.embeddings[embedding_type]
                vectorizer_name, vectorizer_factory, vectorizer_params = embedding_desc
                vectorizer_kwargs_list = self.populate_kwargs({}, vectorizer_params)
                for vectorizer_kwargs in vectorizer_kwargs_list:
                    print("\t\tOn top of %s(%s)" % (vectorizer_name, vectorizer_kwargs), flush=True)

                    vectors_file_name = embedding_type + '_' + self.str_kwargs(vectorizer_kwargs) + '.pkl'
                    vectors_full_path = os.path.join(self.vectors_full_path, dataset_type, vectors_file_name)
                    with open(vectors_full_path, 'rb') as fp:
                        vectors = pickle.load(fp)

                    vectorizer_time = df_embeddings.query(
                        'dataset==\'%s\' & embedding==\'%s\' & vectors_file_name==\'%s\'' %
                        (dataset_name, embedding_type, vectors_file_name))['vectorizer_time'].item()

                    clustering_algorithm_name = algorithm_name[1] \
                        if isinstance(algorithm_name, tuple) else algorithm_name
                    algorithm_display_name, algorithm_factory, algorithm_params = \
                        self.algorithms[clustering_algorithm_name]

                    algorithm_kwargs_list = self.populate_kwargs({
                        'n_clusters': dataset.n_clusters}, algorithm_params)

                    for algorithm_kwargs in algorithm_kwargs_list:
                        with tqdm(range(self.n_runs), file=sys.stdout) as t:
                            for i in t:
                                algorithm_kwargs['random_state'] = self.run_seeds[i]
                                algorithm_display_params = algorithm_kwargs.copy()
                                for param in self.hidden_params:
                                    algorithm_display_params.pop(param, None)
                                t.set_description("\t\t\t%s(%s)" % (algorithm_display_name,
                                                                    str(algorithm_display_params)))
                                t0 = time()
                                algorithm_instance = algorithm_factory(**algorithm_kwargs)
                                algorithm_instance.fit(vectors)
                                algorithm_time = time() - t0
                                results.append({
                                    'dataset': dataset_name,
                                    'n_samples': dataset.n_samples,
                                    'n_clusters': dataset.n_clusters,
                                    'algorithm': algorithm_full_name,
                                    'algorithm_params': str(algorithm_display_params),
                                    'vectorizer_name': vectorizer_name,
                                    'vectorizer_params': vectorizer_kwargs,
                                    'vectorizer_time': vectorizer_time,
                                    'run': i,
                                    'algorithm_time': algorithm_time,
                                    'total_time': algorithm_time + vectorizer_time,
                                    'predicted_labels': algorithm_instance.labels_.tolist(),
                                    'gold_labels': dataset.target.tolist()
                                })

        # create a dataframe
        df = pd.DataFrame(results)

        # append to previous runs if exist
        if os.path.exists(self.predictions_full_path):
            df_prev = pd.read_csv(self.predictions_full_path)
            df_prev.drop(df_prev.filter(regex="Unnamed"), axis=1, inplace=True)
            df = pd.concat([df_prev, df], ignore_index=True)

        # store the metadata of the setups
        df.to_csv(self.predictions_full_path)

    @classmethod
    def classification_scores(cls, row):
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

    def compute_metrics(self):
        if not os.path.exists(self.metrics_full_path):
            df = pd.read_csv(self.predictions_full_path)
            print("Processing %d predictions" % df.shape[0])

            print("Parsing string lists of gold labels")
            df['gold_labels'] = df['gold_labels'].progress_apply(literal_eval)

            print("Parsing string lists of predicted labels")
            df['predicted_labels'] = df['predicted_labels'].progress_apply(literal_eval)

            print("Evaluating predictions")
            df = df.progress_apply(lambda row: self.classification_scores(row), axis=1)

            print("Dropping unneeded columns")
            df = df.drop(labels=['gold_labels', 'predicted_labels'], axis=1)

            df.to_csv(self.metrics_full_path)

    @classmethod
    def bootstrap_ci95(cls, series):
        mean = series.mean()
        if np.allclose(series, series.mean()):
            ci = mean, mean
        else:
            ci = bootstrap((series,), np.mean,
                           random_state=random.randint(0, 1024 ** 2)).confidence_interval
        return mean, ci[0], ci[1]

    @classmethod
    def split_ci_values(cls, row):
        for measure, (mean, low, high) in row.items():
            row[measure] = mean
            row[measure + '_low'] = low
            row[measure + '_high'] = high
        return row

    def aggregate_runs(self):
        if not os.path.exists(self.aggregated_full_path):
            df = pd.read_csv(self.metrics_full_path)
            measures = df.columns[df.dtypes == np.float64]
            df = df.groupby(['dataset', 'algorithm', 'algorithm_params',
                             'vectorizer_name', 'vectorizer_params']).agg(
                {measure: self.bootstrap_ci95 for measure in measures})
            df = df[measures].apply(self.split_ci_values, axis=1)
            df.to_csv(self.aggregated_full_path)

    @classmethod
    def extract_params(cls, param, row):
        for key, value in literal_eval(row[param]).items():
            row[key] = value
        return row

    def create_plots(self):
        df = pd.read_csv(self.aggregated_full_path)

        # sort by the parameters for better arrangement in charts
        df = df.apply(lambda row: self.extract_params('algorithm_params', row), axis=1)
        df = df.apply(lambda row: self.extract_params('vectorizer_params', row), axis=1)

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

        df = self.sort_df(df)

        datasets = [dataset for dataset in df['dataset'].unique()]

        setup_groups = df.groupby(['algorithm', 'vectorizer_name',
                                   'algorithm_params', 'vectorizer_params'], sort=False)

        ind_labels = np.arange(len(datasets)) * (1 + bar_width)  # the x locations for the groups
        ind_values = ind_labels - bar_width * (len(setup_groups) / 2 - 0.5)

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
                    # df_setup = df_setup.sort_values(by=['stop_words']).reset_index(drop=True)
                    mean_values = df_setup[measure]
                    ci_values = df_setup[[measure + '_low', measure + '_high']].T
                    error_bar = (ci_values - mean_values).abs()
                    rects = ax.bar(ind_values + setup_id * bar_width,
                                   mean_values, yerr=error_bar.to_numpy(),
                                   width=bar_width, capsize=3, ecolor='black',
                                   color=colors[setup_id % len(colors)])

                    if log_scale:
                        rotate = df_setup['vectorizer_name'].iloc[0] != self.embeddings['sbert'][0] \
                            if 'sbert' in self.embeddings else True
                        self.autolabel(ax, rects, rotate)

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

            fig.savefig(os.path.join(self.figures_full_path, f'fig_{"_".join(group)}.pdf'),
                        bbox_inches='tight', dpi=300)

    def sort_df(self, df):
        orders = {
            'algorithm': {self.algorithms[a][0]: i for i, a in enumerate(self.algorithm_view_order)},
            'vectorizer_name': {self.embeddings[e][0]: i for i, e in enumerate(self.embedding_view_order)}
        }
        df = df.sort_values(by=['algorithm', 'vectorizer_name'],
                            key=lambda x: x.map(orders[x.name]))
        return df

    @classmethod
    def format_seconds(cls, seconds):
        seconds = int(round(seconds))
        m, s = divmod(seconds, 60)
        if m > 60:
            h, m = divmod(m, 60)
            return f'{h:02d}:{m:02d}:{round(s):02d}'
        else:
            return f'{m:02d}:{round(s):02d}'

    @classmethod
    def autolabel(cls, ax, rects, rotate):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            value = cls.format_seconds(height)
            rotation = 45 if rotate else 0
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, value,
                    # '%.2f' % height,
                    ha='center', va='bottom', fontsize=9, rotation=rotation)

    def create_results_table(self):
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

        df = pd.read_csv(self.aggregated_full_path)
        df = self.sort_df(df)

        for table in sets:
            metrics = table['metrics']
            columns = table['columns']
            caption = table['caption']
            label = table['label']
            wide = table['wide']

            columns.update(metrics)

            opening, fields_spec, ending, columns_names = self.prepare_table(columns, metrics, caption, label, wide)

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
                        metric_latex = f'{self.format_seconds(row[metric])}' \
                            if metric in time_metrics else f'{row[metric]:.2f}'
                        if bolds[metric][i]:
                            metric_latex = f"\\textbf{{" + metric_latex + f"}}"
                        metrics_latex.append(metric_latex)
                    body += " & ".join(metrics_latex)
                    body += " \\\\\n"
                body += "\\hline\n"
            latex = f'{opening}{fields_spec}\n{columns_names}\n{body}\n{ending}'
            with open(os.path.join(self.tables_full_path, table['name']), "wt") as f:
                f.write(latex)

    def create_datasets_table(self):

        fields = {
            'n_samples': '\\# Texts',
            'word_count_mean': '\\# Words',
            'n_clusters': '\\# Classes',
        }
        columns = {
            'dataset': 'Dataset'
        }
        columns.update(fields)

        df = pd.read_csv(self.datasets_metadata_full_path)
        df = df.sort_values(by='n_samples_org')

        # 'The column \\textit{\\# Used} indicates the number of texts used in this evaluation. ' \
        caption = 'Benchmark datasets for evaluation. ' \
                  'The column \\textit{\\# Texts} indicates the number of texts in the dataset. ' \
                  'The column \\textit{\\# Words} shows the average text length in terms of ' \
                  'word-count in the dataset, ' \
                  'and \\textit{\\# Classes} shows the number of classes in the dataset.'
        label = 'tab:datasets'

        opening, fields_spec, ending, columns_names = self.prepare_table(columns, fields, caption, label, False)

        body = ''
        for i, row in df.iterrows():
            values = [int(round(row[col])) if isinstance(row[col], float) else row[col] for col in columns]
            values = [f'{v:,}' if isinstance(v, int) else v for v in values]
            body += " & ".join(values)
            body += " \\\\\n"
            body += "\\hline\n"

        latex = f'{opening}{fields_spec}\n{columns_names}\n{body}\n{ending}'
        with open(os.path.join(self.tables_full_path, 'table_datasets.tex'), "wt") as f:
            f.write(latex)

    @classmethod
    def prepare_table(cls, columns, fields, caption, label, wide):
        wide_opening = f'\\begin{{adjustwidth}}{{-\\extralength}}{{0cm}}\n' if wide else ''
        wide_ending = f'\\end{{adjustwidth}}\n' if wide else ''
        opening = f"\\begin{{table}}[]\n\\caption{{{caption}}}\n\\centering\n{wide_opening}\\begin{{tabular}}"
        ending = f"\\end{{tabular}}\n{wide_ending}\\label{{{label}}}\n\\end{{table}}"
        fields_spec = f"{{|" + " | ".join(["l"] * (len(columns) - len(fields)) + ["c"] * len(fields)) + f"|}}\n"
        columns_names = "\\hline\n" + " & ".join(
            map(lambda x: '\\textbf{' + x + '}', columns.values())) + "\\\\\\hline\n"
        return opening, fields_spec, ending, columns_names

    @staticmethod
    def populate_kwargs(base_params, params):
        param_sets = [base_params]
        for param, values in params.items():
            new_param_sets = []
            for param_set in param_sets:
                for value in values:
                    new_param_set = param_set.copy()
                    new_param_set[param] = value
                    new_param_sets.append(new_param_set)
            param_sets = new_param_sets
        return param_sets

    @staticmethod
    def str_kwargs(d):
        return '_'.join([(key + '-' + str(value)) for key, value in d.items()])

    @staticmethod
    def get_system_desc():
        return {
            'machine': platform.machine(),
            'version': platform.version(),
            'platform': platform.platform(),
            'uname': platform.uname(),
            'system': platform.system(),
            'processor': platform.processor(),
            'ram': str(round(psutil.virtual_memory().total / (1024.0 ** 3))) + " GB",
            'cpus_physical': psutil.cpu_count(logical=False),
            'cpus_logical': psutil.cpu_count(logical=True)
        }

    def prepare(self):
        self.prepare_dirs_model()
        self.read_datasets()
        self.vectorize()
        self.record_hardware()

    def cluster(self):
        self.record_setups()
        self.cluster_setups()

    def evaluate(self):
        self.compute_metrics()
        self.aggregate_runs()
        self.create_plots()
        self.create_results_table()
        self.create_datasets_table()

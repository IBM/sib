# © Copyright IBM Corporation 2020.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import os
from time import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

from sib import SIB
from datasets import fetch_20ng, fetch_bbc, fetch_agnews, fetch_dbpedia


def main():

    # determine the number of runs per algorithm, the number of random inits per run,
    # and the number of cores to use in each run
    n_runs = 10
    n_init = 10
    n_jobs = 10

    # the datasets to use: (name, factory)
    datasets = [
        ('20 News Groups', lambda: fetch_20ng()),
        ('DBPedia', lambda: fetch_dbpedia()),
        ('BBC News', lambda: fetch_bbc()),
        ('AG NEWS', lambda: fetch_agnews()),
    ]

    # the text vectorizers to use: (name, factory)
    vectorizers = {
        'count': lambda: CountVectorizer(max_features=10000),
        'tfidf': lambda: TfidfVectorizer(max_features=10000)
    }

    # the algorithms: {short name: (display name, factory)}
    algorithms = {
        'sib':      ('sIB', lambda x: SIB(n_clusters=x, n_init=n_init, n_jobs=n_jobs, max_iter=15)),
        'kmeans':   ('K-Means', lambda x: KMeans(n_clusters=x, n_init=n_init, n_jobs=n_jobs, algorithm='full')),
        'spectral': ('Spectral', lambda x: SpectralClustering(n_clusters=x, assign_labels="discretize", n_jobs=n_jobs)),
        'lda':      ('LDA', lambda x: LatentDirichletAllocation(n_components=x, n_jobs=n_jobs)),
        'agglomer': ('Agglomer.', lambda x: AgglomerativeClustering(n_clusters=x)),
    }

    # the setups to run: (vectorizer name, vectors manipulator name, algorithms list)
    setups = [
        ('count', '', ['sib']),
        ('tfidf', '', ['kmeans']),
        ('count', 'lda', ['kmeans', 'spectral', 'agglomer'])
    ]

    # certain algorithms do not scale well enough to support large datasets
    exclude_list = {
        'DBPedia': ['spectral', 'agglomer'],
        'AG NEWS': ['spectral', 'agglomer']
    }

    results = {'metrics': [], 'time': []}
    dataset_desc = {}

    # loop over the datasets
    for dataset_name, dataset_fetcher in datasets:
        print("Reading dataset: %s ..." % dataset_name, end=" ", flush=True)
        t1 = time()
        dataset = dataset_fetcher()
        print("done in %.3f sec." % (time() - t1), end=" ", flush=True)
        texts = dataset.data
        gold_labels = dataset.target
        n_clusters = np.unique(gold_labels).shape[0]
        print("%d samples, %d classes" % (len(gold_labels), n_clusters))
        dataset_desc[dataset_name] = (len(gold_labels), n_clusters)

        # prepare the required vector types based on the setups
        vectors_dict = {}
        vector_types = set([setup[0] for setup in setups])
        for vectors_type in vector_types:
            vectorizer = vectorizers[vectors_type]()
            vectors = vectorizer.fit_transform(texts)
            vectors_dict[vectors_type] = vectors

        # iterate over the algorithms and cluster
        for vectors_type, vectors_manipulator_name, algorithms_list in setups:
            for i in range(n_runs):

                # get the vectors for the current setup
                vectors = vectors_dict[vectors_type].copy()
                if vectors_manipulator_name != '':
                    vectors_manipulator_display_name = algorithms[vectors_manipulator_name][0]
                    print("\tApplying %s..." % vectors_manipulator_display_name, end=" ", flush=True)
                    t1 = time()
                    vectors_manipulator = algorithms[vectors_manipulator_name][1](n_clusters)
                    vectors = vectors_manipulator.fit_transform(vectors)
                    vectors_manipulation_time = time() - t1
                    print("done in %.3f sec." % vectors_manipulation_time)
                    indent = "\t\t"
                else:
                    vectors_manipulator_display_name = None
                    vectors_manipulation_time = 0
                    indent = "\t"

                # run the algorithms in the current setup
                for algorithm_name in algorithms_list:
                    algorithm_display_name = algorithms[algorithm_name][0]
                    if dataset_name in exclude_list.keys() and algorithm_name in exclude_list[dataset_name]:
                        print(indent + "Skipping %s due to scalability limitations" % algorithm_display_name)
                        continue
                    print(indent + "Clustering with: %s..." % algorithm_display_name, end=" ", flush=True)
                    t1 = time()
                    algorithm = algorithms[algorithm_name][1](n_clusters)
                    algorithm.fit(vectors)
                    labels = algorithm.labels_.copy()
                    algorithm_time = time() - t1
                    print("done in %.3f sec." % algorithm_time, end=" ", flush=True)
                    sub_setup_name = (vectors_manipulator_display_name + " + " if vectors_manipulator_display_name
                                      else '') + algorithm_display_name

                    # calc metrics and collect data
                    ami = metrics.adjusted_mutual_info_score(gold_labels, labels)
                    ari = metrics.adjusted_rand_score(gold_labels, labels)
                    print("ami: %.3f, ari: %.3f" % (ami, ari))
                    results['metrics'].extend([
                        {'dataset': dataset_name, 'algorithm': sub_setup_name, 'metric': 'ami', 'score': ami},
                        {'dataset': dataset_name, 'algorithm': sub_setup_name, 'metric': 'ari', 'score': ari}])
                    results['time'].append({'dataset': dataset_name, 'algorithm': sub_setup_name, 'metric': 'time',
                                            'seconds': vectors_manipulation_time + algorithm_time})

    # save report
    report_path = os.path.join('output', 'ex4')
    if not os.path.exists(report_path):
        os.makedirs(report_path)

    plot_desc = {
        'metrics': (0.0, 1.0, 0.1, 'score', 'Score', 'Clustering Quality', False),
        'time': (0, 7.0, 0.5, 'seconds', 'Seconds (log scale)', 'Clustering Time', True)
    }

    p = sns.color_palette(['#0070ff', '#71A3E3', '#cccccc'])
    for result_name, result_entries in results.items():
        # create a dataframe from the results and save as a csv
        df = pd.DataFrame(result_entries)
        df.to_csv(os.path.join(report_path, result_name + "_results.csv"))

        # extract settings on the current result type
        min_y_scale, max_y_scale, y_scale_step, y_value, y_label, main_title, log_scale = plot_desc[result_name]

        # apply log to values to compress very high values
        if log_scale:
            df[y_value] = np.log(df[y_value] + 1)
        sns.set_theme(style="whitegrid")

        # create the plot
        g = sns.catplot(x="algorithm", y=y_value,
                        hue="metric", col="dataset", col_wrap=2,
                        data=df, kind="bar", palette=p,
                        height=8, aspect=1, capsize=.1)

        # set the y ticks
        if min_y_scale is not None and max_y_scale is not None:
            g.set(yticks=np.arange(min_y_scale, max_y_scale + y_scale_step, y_scale_step))

        # set font size for x and y axes
        g.set_ylabels(y_label, fontsize=20, labelpad=14)
        for ax in g.axes.flatten():
            ax.tick_params(labelbottom=True)
            ax.set_xlabel("Algorithm", labelpad=14, fontsize=20)
            ax.set_title(ax.get_title(), fontsize=20)
            dataset_name = ax.get_title().split(" = ")[1]
            n_samples, n_classes = dataset_desc[dataset_name]
            new_title = "%s\n(%.1fk texts, %d classes)" % (dataset_name, n_samples / 1000, n_classes)
            ax.set_title(new_title, fontsize=20)

        # spacing between sub-plots and between plots and main title
        plt.subplots_adjust(top=0.9, hspace=0.3, wspace=0.2)

        # set the legend
        g.legend.set_title('')
        plt.setp(g.legend.get_texts(), fontsize='18')  # for legend text
        plt.setp(g.legend.get_title(), fontsize='20')  # for legend title

        # set the main title
        g.fig.suptitle(main_title, fontsize=24)

        # save plot
        g.savefig(os.path.join(report_path, result_name + "_results.svg"), bbox_inches='tight')
        g.savefig(os.path.join(report_path, result_name + "_results.png"), bbox_inches='tight', dpi=240)


if __name__ == '__main__':
    main()

import sklearn
from sklearn import metrics
import numpy as np
import scipy.stats as st
import pandas as pd
import scipy


class predsummary_2class:
    '''
    predlabels and gtlabels are strings with values 0 for negative label and 1 for positive label
    '''
    #
    def __init__(self, predlabels, gtlabels, predscores=[]):
        self.predscores = predscores
        self.predlabels = predlabels
        self.gtlabels = gtlabels
        self.st = scipy.stats
    #
    def get_confusion_matrix(
        self, predlabels, gtlabels,
        labels=None, sample_weight=None, normalize=None
    ):
        '''
        when labels are 0 and 1, then the location of TP and TN are interchanged in the array returned by confusion_matrix in sklearn
        '''
        c_matrix = metrics.confusion_matrix(
            y_pred=predlabels,
            y_true=gtlabels,
            labels=labels,
            sample_weight=sample_weight,
            normalize=normalize
        )
        # if any(np.unique(predlabels) == 0):
            # out_c_matrix = np.array( [[c_matrix[1,1], c_matrix[0,1]], [c_matrix[1,0], c_matrix[0,0]]])
        # out_c_matrix = np.reshape(
        #     np.array(
        #         [
        #             c_matrix[1,1], c_matrix[0,1],
        #             c_matrix[1,0], c_matrix[0,0]
        #         ]
        #     ),
        #     [2,2]
        # )
        # if not any(np.unique(predlabels) == 0):
        #     out_c_matrix = c_matrix
        out_dict = dict()
        tn, fp, fn, tp =c_matrix.ravel()
        out_dict['cmatrix'] = c_matrix
        out_dict['tp'] = tp
        out_dict['fp'] = fp
        out_dict['fn'] = fn
        out_dict['tn'] = tn
        return out_dict
    #
    def se_wilson(
        self, n_success, n_total,
        alpha=0.05, continuity=False
    ):
        '''
        Wilson confidence intervals formulas from here - https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/PASS/Confidence_Intervals_for_One-Sample_Sensitivity.pdf
        '''

        phat = n_success / n_total
        z1minusalphaby2 = self.st.norm.ppf(
            1 - (alpha / 2)
        )
        if not continuity:
            numerator_lhs = (2 * n_total * phat) + (z1minusalphaby2 ** 2)
            numerator_rhs = z1minusalphaby2 * (
                np.sqrt(
                    (z1minusalphaby2 ** 2) + (4 * n_total * phat * (1-phat))
                )
            )
            denominator = 2 * (n_total + (z1minusalphaby2 ** 2))
            lci = (numerator_lhs - numerator_rhs) / denominator
            uci = (numerator_lhs + numerator_rhs) / denominator
            wilson_dict = dict()
            wilson_dict['lci'] = lci
            wilson_dict['uci'] = uci
        return wilson_dict
    #
    def compute_recall_sensitivity(self, in_tp, in_fn, alpha=0.05):
        recall_est = in_tp / (in_tp + in_fn+1e-6)
        recall_se = self.se_wilson(in_tp, n_total=(in_tp + in_fn+1e-6), alpha=alpha)
        recall_estsedict = dict()
        recall_estsedict['est'] = recall_est
        recall_estsedict['lci'] = recall_se['lci']
        recall_estsedict['uci'] = recall_se['uci']
        return recall_estsedict
    #
    def compute_specificity(self, in_tn, in_fp, alpha=0.05):
        specificity_est = in_tn / (in_tn + in_fp+1e-6)
        specificity_se = self.se_wilson(
            in_tn, n_total=(in_tn + in_fp+1e-6), alpha=alpha
        )
        specificity_estsedict = dict()
        specificity_estsedict['est'] = specificity_est
        specificity_estsedict['lci'] = specificity_se['lci']
        specificity_estsedict['uci'] = specificity_se['uci']
        return specificity_estsedict
    #
    def compute_ppv(self, in_tp, in_fp, alpha=0.05):
        ppv_est = in_tp / (in_tp + in_fp+1e-6)
        ppv_se = self.se_wilson(
            in_tp, n_total=(in_tp + in_fp+1e-6), alpha=alpha
        )
        ppv_estsedict = dict()
        ppv_estsedict['est'] = ppv_est
        ppv_estsedict['lci'] = ppv_se['lci']
        ppv_estsedict['uci'] = ppv_se['uci']
        return ppv_estsedict
    #
    def compute_npv(self, in_tn, in_fn, alpha=0.05):
        npv_est = in_tn / (in_tn + in_fn+1e-6)
        npv_se = self.se_wilson(
            in_tn, n_total=(in_tn + in_fn+1e-6), alpha=alpha
        )
        npv_estsedict = dict()
        npv_estsedict['est'] = npv_est
        npv_estsedict['lci'] = npv_se['lci']
        npv_estsedict['uci'] = npv_se['uci']
        return npv_estsedict
    #
    def compute_accuracy(self, in_tp, in_fp, in_fn, in_tn, alpha=0.05):
        accuracy_est = (in_tp + in_tn) / (in_tp + in_fp + in_fn + in_tn+1e-6)
        accuracy_se = self.se_wilson(
            (in_tp + in_tn), n_total=(in_tp + in_fp + in_fn + in_tn+1e-6),
            alpha=alpha
        )
        accuracy_se
        accuracy_estsedict = dict()
        accuracy_estsedict['est'] = accuracy_est
        accuracy_estsedict['lci'] = accuracy_se['lci']
        accuracy_estsedict['uci'] = accuracy_se['uci']
        return accuracy_estsedict
    #
    '''
    AUC code sources:
    Original code: https://github.com/yandexdataschool/roc_comparison/blob/master/compare_auc_delong_xu.py
    Code for method to compute CI: https://github.com/RaulSanchezVazquez/roc_curve_with_confidence_intervals/blob/master/auc_delong_xu.py
    '''
    # AUC comparison adapted from
    # https://github.com/Netflix/vmaf/
    def compute_midrank(self, x):
        """Computes midranks.
        Args:
        x - a 1D numpy array
        Returns:
        array of midranks
        """
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=np.float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5*(i + j - 1)
            i = j
        T2 = np.empty(N, dtype=np.float)
        # Note(kazeevn) +1 is due to Python using 0-based indexing
        # instead of 1-based in the AUC formula in the paper
        T2[J] = T + 1
        return T2
    #
    def fastDeLong(self, predictions_sorted_transposed, label_1_count):
        """
        The fast version of DeLong's method for computing the covariance of
        unadjusted AUC.
        Args:
        predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
            sorted such as the examples with label "1" are first
        Returns:
        (AUC value, DeLong covariance)
        Reference:
        @article{sun2014fast,
        title={Fast Implementation of DeLong's Algorithm for
                Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
        author={Xu Sun and Weichao Xu},
        journal={IEEE Signal Processing Letters},
        volume={21},
        number={11},
        pages={1389--1393},
        year={2014},
        publisher={IEEE}
        }
        """
        # Short variables are named as they are in the paper
        m = label_1_count
        n = predictions_sorted_transposed.shape[1] - m
        positive_examples = predictions_sorted_transposed[:, :m]
        negative_examples = predictions_sorted_transposed[:, m:]
        k = predictions_sorted_transposed.shape[0]

        tx = np.empty([k, m], dtype=np.float)
        ty = np.empty([k, n], dtype=np.float)
        tz = np.empty([k, m + n], dtype=np.float)
        for r in range(k):
            tx[r, :] = self.compute_midrank(positive_examples[r, :])
            ty[r, :] = self.compute_midrank(negative_examples[r, :])
            tz[r, :] = self.compute_midrank(predictions_sorted_transposed[r, :])
        aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
        v01 = (tz[:, :m] - tx[:, :]) / n
        v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
        sx = np.cov(v01)
        sy = np.cov(v10)
        delongcov = sx / m + sy / n
        return aucs, delongcov
    #
    def calc_pvalue(self, aucs, sigma):
        """Computes log(10) of p-values.
        Args:
        aucs: 1D array of AUCs
        sigma: AUC DeLong covariances
        Returns:
        log10(pvalue)
        """
        l = np.array([[1, -1]])
        z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
        return np.log10(2) + self.st.norm.logsf(z, loc=0, scale=1) / np.log(10)
    #
    def compute_ground_truth_statistics(self, ground_truth):
        assert np.array_equal(np.unique(ground_truth), [0, 1])
        order = (-ground_truth).argsort()
        label_1_count = int(ground_truth.sum())
        return order, label_1_count
    #
    def delong_roc_variance(self, ground_truth, predictions):
        """
        Computes ROC AUC variance for a single set of predictions
        Args:
        ground_truth: np.array of 0 and 1
        predictions: np.array of floats of the probability of being class 1
        """
        order, label_1_count = self.compute_ground_truth_statistics(ground_truth)
        predictions_sorted_transposed = predictions[np.newaxis, order]
        aucs, delongcov = self.fastDeLong(predictions_sorted_transposed, label_1_count)
        assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
        return aucs[0], delongcov
    #
    def delong_roc_test(self, ground_truth, predictions_one, predictions_two):
        """
        Computes log(p-value) for hypothesis that two ROC AUCs are different
        Args:
        ground_truth: np.array of 0 and 1
        predictions_one: predictions of the first model,
            np.array of floats of the probability of being class 1
        predictions_two: predictions of the second model,
            np.array of floats of the probability of being class 1
        """
        order, label_1_count = self.compute_ground_truth_statistics(ground_truth)
        predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
        aucs, delongcov = self.fastDeLong(predictions_sorted_transposed, label_1_count)
        return self.calc_pvalue(aucs, delongcov)
    #
    def auc_ci_Delong(self, y_true, y_scores, alpha=0.95):
        """AUC confidence interval via DeLong.

        Computes de ROC-AUC with its confidence interval via delong_roc_variance

        References
        -----------
            See this `Stack Overflow Question
            <https://stackoverflow.com/questions/19124239/scikit-learn-roc-curve-with-confidence-intervals/53180614#53180614/>`_
            for further details

        Examples
        --------

            y_scores = np.array(
                [0.21, 0.32, 0.63, 0.35, 0.92, 0.79, 0.82, 0.99, 0.04])
            y_true = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0])

            auc, auc_var, auc_ci = auc_ci_Delong(y_true, y_scores, alpha=.95)

            np.sqrt(auc_var) * 2
            max(auc_ci) - min(auc_ci)

            print('AUC: %s' % auc, 'AUC variance: %s' % auc_var)
            print('AUC Conf. Interval: (%s, %s)' % tuple(auc_ci))

            Out:
                AUC: 0.8 AUC variance: 0.028749999999999998
                AUC Conf. Interval: (0.4676719375452081, 1.0)


        Parameters
        ----------
        y_true : list
            Ground-truth of the binary labels (allows labels between 0 and 1).
        y_scores : list
            Predicted scores.
        alpha : float
            Default 0.95

        Returns
        -------
            auc : float
                AUC
            auc_var : float
                AUC Variance
            auc_ci : tuple
                AUC Confidence Interval given alpha

        """

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        # Get AUC and AUC variance
        auc, auc_var = self.delong_roc_variance(
            ground_truth=y_true,
            predictions=y_scores)
        auc_std = np.sqrt(auc_var)

        # Confidence Interval
        lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
        lower_upper_ci = self.st.norm.ppf(
            lower_upper_q,
            loc=auc,
            scale=auc_std)
        lower_upper_ci[lower_upper_ci > 1] = 1
        return auc, auc_var, lower_upper_ci
#
def presentpredsummary(
    in_methodname, in_dsetname, 
    in_predlabels, in_gtlabels, in_predscores, 
    csvfilepath=None,
    accuracy=True, recall=True, specificity=True,
    ppv=True, npv=True, auc=True,
    writecsv=True):
    in_predsummary = predsummary_2class(
        predlabels=in_predlabels,
        gtlabels=in_gtlabels,
        predscores=in_predscores
    )
    cm = in_predsummary.get_confusion_matrix(
        predlabels=in_predlabels,
        gtlabels=in_gtlabels
    )
    if accuracy:
        est_accuracy = in_predsummary.compute_accuracy(
            cm['tp'], cm['fp'], cm['fn'], cm['tn']
        )
        out_accuracy = "{:.2f}".format(est_accuracy['est']) + ' (' + "{:.2f}".format(est_accuracy['lci']) + ' to ' + "{:.2f}".format(est_accuracy['uci']) + ')'
    else:
        out_accuracy = 'not computed'
    #
    if recall:
        est_recall = in_predsummary.compute_recall_sensitivity(
            cm['tp'], cm['fn']
        )
        out_recall = "{:.2f}".format(est_recall['est']) + ' (' + "{:.2f}".format(est_recall['lci']) + ' to ' + "{:.2f}".format(est_recall['uci']) + ')'
    else:
        out_recall = 'not computed'
    #
    if specificity:
        est_specificity = in_predsummary.compute_specificity(
            cm['tn'], cm['fp']
        )
        out_specificity = "{:.2f}".format(est_specificity['est']) + ' (' + "{:.2f}".format(est_specificity['lci']) + ' to ' + "{:.2f}".format(est_specificity['uci']) + ')'
    else:
        out_specificity = 'not computed'
    #
    if ppv:
        est_ppv = in_predsummary.compute_ppv(
            cm['tp'], cm['fp']
        )
        out_ppv = "{:.2f}".format(est_ppv['est']) + ' (' + "{:.2f}".format(est_ppv['lci']) + ' to ' + "{:.2f}".format(est_ppv['uci']) + ')'
    else:
        out_ppv = 'not computed'
    #
    if npv:
        est_npv = in_predsummary.compute_npv(
            cm['tn'], cm['fn']
        )
        out_npv = "{:.2f}".format(est_npv['est']) + ' (' + "{:.2f}".format(est_npv['lci']) + ' to ' + "{:.2f}".format(est_npv['uci']) + ')'
    else:
        out_npv = 'not computed'
    #
    if auc:
        est_auc, var_auc, ci_auc = in_predsummary.auc_ci_Delong(
            in_gtlabels, in_predscores
        )
        out_auc = "{:.2f}".format(est_auc) + ' (' + "{:.2f}".format(ci_auc[0]) + ' to ' + "{:.2f}".format(ci_auc[1]) + ')'
    else:
        out_auc = 'not computed'
    #
    out_ls = [
        in_methodname,
        in_dsetname,
        out_accuracy,
        out_recall,
        out_specificity,
        out_ppv,
        out_npv,
        out_auc
    ]
    out_lsdf = pd.DataFrame(out_ls)
    out_df = out_lsdf.T
    out_df.rename(
        columns={
            0: 'Method',
            1: 'Experiment',
            2: 'Accuracy',
            3: 'Sensitivity',
            4: 'Specificity',
            5: 'PPV',
            6: 'NPV',
            7: 'AUC'
        },
        inplace=True
    )
    # out_df.columns.values = [
    #         'Method',
    #         'Experiment'
    #         'Accuracy',
    #         'Sensitivity',
    #         'Specificity',
    #         'PPV', 'NPV', 'AUC'
    #     ]
    if writecsv:
        out_df.to_csv(csvfilepath, index=False)
    #
    return out_df
#
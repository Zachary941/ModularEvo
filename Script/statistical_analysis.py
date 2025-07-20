import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class StatisticalAnalyzer:
    """Statistical analyzer class"""

    def __init__(self):
        # Experimental data
        self.data = {
            # codebert
            'WA': [93.64, 94.08, 94.16],
            'TA': [94.24, 95.41, 96.07],
            'DARE': [96.73, 97.02, 96.74],
            'TIES-Merging': [95.51, 95.76, 95.59],
            'ModularEvo': [98.70, 98.55, 98.76]
        }

        self.results = {}

    def calculate_basic_statistics(self) -> Dict:
        """Calculate basic statistics: mean and standard deviation"""
        print("=" * 60)
        print("Basic Statistics Analysis")
        print("=" * 60)

        basic_stats = {}

        for method, values in self.data.items():
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)  # Use sample standard deviation

            basic_stats[method] = {
                'values': values,
                'mean': mean_val,
                'std': std_val,
                'n': len(values)
            }

            print(f"{method}:")
            print(f"  Data: {values}")
            print(f"  Mean: {mean_val:.4f}")
            print(f"  Standard Deviation: {std_val:.4f}")
            print(f"  Sample Size: {len(values)}")
            print()

        self.results['basic_stats'] = basic_stats
        return basic_stats

    def perform_statistical_tests(self) -> Dict:
        """Perform statistical significance testing"""
        print("=" * 60)
        print("Statistical Significance Testing (t-test)")
        print("=" * 60)

        modular_evo = self.data['ModularEvo']
        dare = self.data['DARE']
        ties = self.data['TIES-Merging']
        wa = self.data['WA']
        ta = self.data['TA']

        # ModularEvo vs DARE
        t_stat_dare, p_value_dare = stats.ttest_ind(modular_evo, dare)

        # ModularEvo vs TIES-Merging
        t_stat_ties, p_value_ties = stats.ttest_ind(modular_evo, ties)

        # ModularEvo vs WA
        t_stat_wa, p_value_wa = stats.ttest_ind(modular_evo, wa)

        # ModularEvo vs TA
        t_stat_ta, p_value_ta = stats.ttest_ind(modular_evo, ta)

        # DARE vs TIES-Merging
        t_stat_dare_ties, p_value_dare_ties = stats.ttest_ind(dare, ties)

        test_results = {
            'ModularEvo_vs_DARE': {
                't_statistic': t_stat_dare,
                'p_value': p_value_dare,
                'significant': p_value_dare < 0.05
            },
            'ModularEvo_vs_TIES': {
                't_statistic': t_stat_ties,
                'p_value': p_value_ties,
                'significant': p_value_ties < 0.05
            },
            'ModularEvo_vs_WA': {
                't_statistic': t_stat_wa,
                'p_value': p_value_wa,
                'significant': p_value_wa < 0.05
            },
            'ModularEvo_vs_TA': {
                't_statistic': t_stat_ta,
                'p_value': p_value_ta,
                'significant': p_value_ta < 0.05
            },
            'DARE_vs_TIES': {
                't_statistic': t_stat_dare_ties,
                'p_value': p_value_dare_ties,
                'significant': p_value_dare_ties < 0.05
            }
        }
        print("ModularEvo vs WA:")
        print(f"  t-statistic: {t_stat_wa:.4f}")
        print(f"  p-value: {p_value_wa:.6f}")
        print(f"  Significant (p<0.05): {'Yes' if p_value_wa < 0.05 else 'No'}")
        print()

        print("ModularEvo vs TA:")
        print(f"  t-statistic: {t_stat_ta:.4f}")
        print(f"  p-value: {p_value_ta:.6f}")
        print(f"  Significant (p<0.05): {'Yes' if p_value_ta < 0.05 else 'No'}")
        print()

        print("ModularEvo vs DARE:")
        print(f"  t-statistic: {t_stat_dare:.4f}")
        print(f"  p-value: {p_value_dare:.6f}")
        print(f"  Significant (p<0.05): {'Yes' if p_value_dare < 0.05 else 'No'}")
        print()

        print("ModularEvo vs TIES-Merging:")
        print(f"  t-statistic: {t_stat_ties:.4f}")
        print(f"  p-value: {p_value_ties:.6f}")
        print(f"  Significant (p<0.05): {'Yes' if p_value_ties < 0.05 else 'No'}")
        print()

        self.results['statistical_tests'] = test_results
        return test_results

    def calculate_effect_size(self) -> Dict:
        """Calculate effect size (Cohen's d)"""
        print("=" * 60)
        print("Effect Size Analysis (Cohen's d)")
        print("=" * 60)

        def cohens_d(group1, group2):
            """Calculate Cohen's d effect size"""
            n1, n2 = len(group1), len(group2)
            s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

            # Pooled standard deviation
            pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))

            # Cohen's d
            d = (np.mean(group1) - np.mean(group2)) / pooled_std
            return d

        def interpret_effect_size(d):
            """Interpret effect size"""
            abs_d = abs(d)
            if abs_d < 0.2:
                return "Small effect"
            elif abs_d < 0.5:
                return "Medium effect"
            elif abs_d < 0.8:
                return "Large effect"
            else:
                return "Very large effect"

        modular_evo = self.data['ModularEvo']
        dare = self.data['DARE']
        ties = self.data['TIES-Merging']
        wa = self.data['WA']
        ta = self.data['TA']

        # Calculate effect sizes
        d_modular_dare = cohens_d(modular_evo, dare)
        d_modular_ties = cohens_d(modular_evo, ties)
        d_modular_wa = cohens_d(modular_evo, wa)
        d_modular_ta = cohens_d(modular_evo, ta)
        d_dare_ties = cohens_d(dare, ties)

        effect_sizes = {
            'ModularEvo_vs_DARE': {
                'cohens_d': d_modular_dare,
                'interpretation': interpret_effect_size(d_modular_dare)
            },
            'ModularEvo_vs_TIES': {
                'cohens_d': d_modular_ties,
                'interpretation': interpret_effect_size(d_modular_ties)
            },
            'ModularEvo_vs_WA': {
                'cohens_d': d_modular_wa,
                'interpretation': interpret_effect_size(d_modular_wa)
            },
            'ModularEvo_vs_TA': {
                'cohens_d': d_modular_ta,
                'interpretation': interpret_effect_size(d_modular_ta)
            },
            'DARE_vs_TIES': {
                'cohens_d': d_dare_ties,
                'interpretation': interpret_effect_size(d_dare_ties)
            }
        }

        print("ModularEvo vs DARE:")
        print(f"  Cohen's d: {d_modular_dare:.4f}")
        print(f"  Effect size: {interpret_effect_size(d_modular_dare)}")
        print()

        print("ModularEvo vs TIES-Merging:")
        print(f"  Cohen's d: {d_modular_ties:.4f}")
        print(f"  Effect size: {interpret_effect_size(d_modular_ties)}")
        print()

        print("ModularEvo vs WA:")
        print(f"  Cohen's d: {d_modular_wa:.4f}")
        print(f"  Effect size: {interpret_effect_size(d_modular_wa)}")
        print()

        print("ModularEvo vs TA:")
        print(f"  Cohen's d: {d_modular_ta:.4f}")
        print(f"  Effect size: {interpret_effect_size(d_modular_ta)}")
        print()

        print("DARE vs TIES-Merging:")
        print(f"  Cohen's d: {d_dare_ties:.4f}")
        print(f"  Effect size: {interpret_effect_size(d_dare_ties)}")
        print()

        self.results['effect_sizes'] = effect_sizes
        return effect_sizes

    def generate_summary_report(self):
        """Generate summary report"""
        print("=" * 60)
        print("Summary Report")
        print("=" * 60)

        # Create DataFrame for display
        summary_data = []
        for method in self.data.keys():
            stats = self.results['basic_stats'][method]
            summary_data.append({
                'Method': method,
                'Experiment 1': stats['values'][0],
                'Experiment 2': stats['values'][1],
                'Experiment 3': stats['values'][2],
                'Mean': f"{stats['mean']:.4f}",
                'Std Dev': f"{stats['std']:.4f}"
            })

        df = pd.DataFrame(summary_data)
        print("Performance Statistics Table:")
        print(df.to_string(index=False))
        print()

        print("Statistical Significance Test Results:")
        for comparison, result in self.results['statistical_tests'].items():
            print(f"  {comparison.replace('_', ' ')}: p = {result['p_value']:.6f} ({'Significant' if result['significant'] else 'Not significant'})")

        print()
        print("Effect Size Analysis:")
        for comparison, result in self.results['effect_sizes'].items():
            print(f"  {comparison.replace('_', ' ')}: Cohen's d = {result['cohens_d']:.4f} ({result['interpretation']})")

        print()
        print("Conclusions:")
        modular_mean = self.results['basic_stats']['ModularEvo']['mean']
        dare_mean = self.results['basic_stats']['DARE']['mean']
        ties_mean = self.results['basic_stats']['TIES-Merging']['mean']

        print(f"1. ModularEvo average performance ({modular_mean:.4f}) outperforms DARE ({dare_mean:.4f}) and TIES-Merging ({ties_mean:.4f})")
        print(f"2. Performance improvement: vs DARE = {modular_mean - dare_mean:.4f}, vs TIES-Merging = {modular_mean - ties_mean:.4f}")

        # Save results to file
        with open('statistical_analysis_results.txt', 'w', encoding='utf-8') as f:
            f.write("ModularEvo vs Baseline Methods Statistical Analysis Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(df.to_string(index=False))
            f.write("\n\nStatistical Significance Testing:\n")
            for comparison, result in self.results['statistical_tests'].items():
                f.write(f"{comparison}: p = {result['p_value']:.6f}\n")
            f.write("\nEffect Size Analysis:\n")
            for comparison, result in self.results['effect_sizes'].items():
                f.write(f"{comparison}: Cohen's d = {result['cohens_d']:.4f} ({result['interpretation']})\n")

        print("\nDetailed results saved to: statistical_analysis_results.txt")

def main():
    """Main function"""
    print("ModularEvo vs Baseline Methods Statistical Analysis")
    print("=" * 60)

    analyzer = StatisticalAnalyzer()

    # Execute analysis
    analyzer.calculate_basic_statistics()
    analyzer.perform_statistical_tests()
    analyzer.calculate_effect_size()
    analyzer.generate_summary_report()

if __name__ == "__main__":
    main()

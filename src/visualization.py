import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_stress_proportion(data):
    total = data['stress_level'].value_counts().values.sum()
    label = ['Rendah', 'Normal', 'Relatif Tinggi']
    colors = sns.color_palette('pastel')

    plt.figure(figsize=(8, 6))
    plt.pie(data['stress_level'].value_counts().values, labels=label, autopct=lambda x: '{:.1f}%\n({:.0f})'.format(x, total*x/100), colors=colors)
    plt.title('Proporsi Level Stress Siswa')
    plt.show()

def plot_correlation_heatmap(data):
    corr = np.corrcoef(data['blood_pressure'], data['stress_level'])
    print("Correlation coefficient: \n", corr)
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr, annot=True, cmap='YlGnBu')
    plt.title('Heatmap Korelasi Pearson')
    plt.show()

def plot_pairplots(data):
    psychologist = data[['anxiety_level', 'self_esteem', 'mental_health_history', 'depression', 'stress_level']]
    fisio = data[['headache', 'blood_pressure', 'sleep_quality', 'breathing_problem', 'stress_level']]
    social = data[['social_support', 'peer_pressure', 'extracurricular_activities', 'bullying', 'stress_level']]
    academic = data[['academic_performance', 'study_load', 'teacher_student_relationship', 'future_career_concerns', 'stress_level']]
    envi = data[['noise_level', 'living_conditions', 'safety', 'basic_needs', 'stress_level']]

    pairplot = sns.pairplot(psychologist, hue="stress_level")
    pairplot.fig.suptitle("Hubungan Faktor Psikologis berdasarkan Level Stres", y=1.02)
    plt.show()

    pairplot = sns.pairplot(fisio, hue="stress_level")
    pairplot.fig.suptitle("Hubungan Faktor Fisiologis berdasarkan Level Stres", y=1.02)
    plt.show()

    pairplot = sns.pairplot(social, hue="stress_level")
    pairplot.fig.suptitle("Hubungan Faktor Sosial berdasarkan Level Stres", y=1.02)
    plt.show()

    pairplot = sns.pairplot(academic, hue="stress_level")
    pairplot.fig.suptitle("Hubungan Faktor Akademik berdasarkan Level Stres", y=1.02)
    plt.show()

    pairplot = sns.pairplot(envi, hue="stress_level")
    pairplot.fig.suptitle("Hubungan Faktor Lingkungan berdasarkan Level Stres", y=1.02)
    plt.show()

from src.data_processing import load_data, clean_data
from src.feature_engineering import extract_features
from src.model_training import split_data, train_model
from src.model_evaluation import evaluate_model, display_confusion_matrix
from src.visualization import plot_stress_proportion, plot_correlation_heatmap, plot_pairplots

def main():
    data = load_data('./data/StressLevelDataset.csv')
    data = clean_data(data)
    
    plot_stress_proportion(data)
    plot_correlation_heatmap(data)
    plot_pairplots(data)

    X, y = extract_features(data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    accuracy, precision, recall, f1, cm = evaluate_model(model, X_test, y_test)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    display_confusion_matrix(cm)

if __name__ == '__main__':
    main()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error

#Global TimeSeriesSplit instance. 7 folds chosen for stability on ~4,000 hourly samples
tscv = TimeSeriesSplit(n_splits=7)

#Runs CV and test evaluation on multiple models and returns a ranked comparison table.
#To identify the strongest candidates worth tuning.

def compare_models(models_dict, X_train, y_train, X_test, y_test, cv):
    results = []

    for name, model in models_dict.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            'Model': name,
            'CV R² (mean)': round(cv_scores.mean(), 3),
            'CV R² (std)': round(cv_scores.std(), 3),
            'Test R²': round(r2_score(y_test, y_pred), 3),
            'Test MAE': round(mean_absolute_error(y_test, y_pred), 3)
        })

    return pd.DataFrame(results).sort_values('Test R²', ascending=False)


# GridSearchCV + test evaluation, returns best model.

def tune_model(model, param_grid, X_train, y_train, X_test, y_test, cv):
    """GridSearchCV + test evaluation, returns best model"""
    grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV R²: {grid_search.best_score_:.3f}")
    print(f"Test R²: {r2_score(y_test, y_pred):.3f}")
    print(f"Test MAE: {mean_absolute_error(y_test, y_pred):.3f}")
    return best_model


#For tree-based models. Shows top N most important features
#Useful for interpreting what the model learned and detecting unexpected dominance

def feature_importance(model, X_train, top_n=15):
    """For tree-based models — shows top N most important features"""
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    return importances.sort_values(ascending=False).head(top_n)


#Visual check: predicted vs actual over the test period

def plot_predictions(model, X_test, y_test):
    """Visual check — predicted vs actual"""
    y_pred = model.predict(X_test)
    plt.figure(figsize=(12, 4))
    plt.plot(y_test.values, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.legend()
    plt.title('Predicted vs Actual - Iron Concentrate %')
    plt.ylabel('Iron Concentrate %')
    plt.show()
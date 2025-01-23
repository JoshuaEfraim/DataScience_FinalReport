import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, mean_squared_error
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

file_path = "turkiye-student-evaluation_generic.csv"  
data = pd.read_csv(file_path)
print(data.head())

missing_values = data.isnull().sum()
print("Checking missing values")
print(missing_values)

correlation_matrix = data.corr()

course_features = [f"Q{i}" for i in range(1, 13)]
instructor_features = [f"Q{i}" for i in range(13, 29)]
targets = ["nb.repeat", "attendance", "difficulty"]


for target in targets:
    print(f"\nCorrelation Analysis for {target.capitalize()}:")
    course_corr = correlation_matrix.loc[course_features, target].sort_values(ascending=False)
    print(f"\nTop Correlations with {target.capitalize()} (Course):")
    print(course_corr.head(5))
    plt.figure(figsize=(8, 6))
    sns.heatmap(course_corr.to_frame(), annot=True, cmap="coolwarm", cbar=False)
    plt.title(f"Correlations of Q1-Q12 with {target.capitalize()} (Course)")
    plt.show()

    instructor_corr = correlation_matrix.loc[instructor_features, target].sort_values(ascending=False)
    print(f"\nTop Correlations with {target.capitalize()} (Instructor):")
    print(instructor_corr.head(5))
    plt.figure(figsize=(8, 8))
    sns.heatmap(instructor_corr.to_frame(), annot=True, cmap="coolwarm", cbar=False)
    plt.title(f"Correlations of Q13-Q28 with {target.capitalize()} (Instructor)")
    plt.show()

print("\nPerforming Network Analysis for Attendance, Repeat, and Difficulty...\n")

outcome_variables = ["nb.repeat", "attendance", "difficulty"]

correlations = data[outcome_variables].corr()
print("Correlation Matrix:")
print(correlations)

G = nx.Graph()

for var in outcome_variables:
    G.add_node(var, label=var)

for i in range(len(outcome_variables)):
    for j in range(i + 1, len(outcome_variables)):  
        weight = correlations.iloc[i, j]
        G.add_edge(outcome_variables[i], outcome_variables[j], weight=weight)

plt.figure(figsize=(8, 6))
pos = nx.circular_layout(G)  
edge_labels = nx.get_edge_attributes(G, 'weight')  

nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=12, font_weight='bold')
nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()}, font_size=10)
plt.title("Network Analysis: Attendance, Repeat, and Difficulty")
plt.show()

print("\nEdge Weights (Correlations):")
for edge in G.edges(data=True):
    print(f"{edge[0]} <--> {edge[1]}: {edge[2]['weight']:.2f}")

degree_centrality = nx.degree_centrality(G)
print("\nDegree Centrality of Outcome Variables:")
for node, centrality in degree_centrality.items():
    print(f"{node}: {centrality:.2f}")

X_combined = data[course_features + instructor_features]  

for target in targets:
    print(f"\nAnalyzing Target: {target.capitalize()}")
    
    X_course = data[course_features]  
    X_instructor = data[instructor_features]  
    
    y = data[target]  
    
    X_train_course, X_test_course, y_train, y_test = train_test_split(X_course, y, test_size=0.2, random_state=42)
    
    X_train_instructor, X_test_instructor, _, _ = train_test_split(X_instructor, y, test_size=0.2, random_state=42)
    
    rf_course = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_course.fit(X_train_course, y_train)
    accuracy_course = accuracy_score(y_test, rf_course.predict(X_test_course))
    feature_importances_course = pd.Series(rf_course.feature_importances_, index=course_features)
    
    print(f"Random Forest Accuracy for {target.capitalize()} (Course Features): {accuracy_course:.2f}")
    
    plt.figure(figsize=(10, 6))
    sorted_importances_course = feature_importances_course.sort_values(ascending=False)
    ax = sorted_importances_course.plot(kind='bar', color='lightgreen')

    for i, value in enumerate(sorted_importances_course):
        plt.text(i, value + 0.002, f"{value:.3f}", ha='center', va='bottom', fontsize=9, rotation=0)

    plt.title(f"Feature Importances for {target.capitalize()} (Course Features - Random Forest)")
    plt.ylabel("Importance Score")
    plt.xlabel("Features")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    rf_instructor = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_instructor.fit(X_train_instructor, y_train)
    accuracy_instructor = accuracy_score(y_test, rf_instructor.predict(X_test_instructor))
    feature_importances_instructor = pd.Series(rf_instructor.feature_importances_, index=instructor_features)
    
    print(f"Random Forest Accuracy for {target.capitalize()} (Instructor Features): {accuracy_instructor:.2f}")
    
    plt.figure(figsize=(10, 6))
    sorted_importances_instructor = feature_importances_instructor.sort_values(ascending=False)
    ax = sorted_importances_instructor.plot(kind='bar', color='skyblue')

    for i, value in enumerate(sorted_importances_instructor):
        plt.text(i, value + 0.002, f"{value:.3f}", ha='center', va='bottom', fontsize=9, rotation=0)

    plt.title(f"Feature Importances for {target.capitalize()} (Instructor Features - Random Forest)")
    plt.ylabel("Importance Score")
    plt.xlabel("Features")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    
    if target == "nb.repeat":
        lr_model = LinearRegression()
        lr_model.fit(X_course, y) 
        lr_predictions = lr_model.predict(X_course)
        mse = mean_squared_error(y, lr_predictions)
        print(f"Linear Regression MSE for {target.capitalize()}: {mse:.2f}")
    
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
    
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)
    svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test))
    print(f"SVM Accuracy for {target.capitalize()} (All Features): {svm_accuracy:.2f}")

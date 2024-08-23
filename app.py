import pandas as pd
import os
from tkinter import Tk, Label, Button, filedialog, messagebox
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

def get_next_file_number(base_path, base_name):
    existing_files = [f for f in os.listdir(base_path) if f.startswith(base_name) and f.endswith('.csv')]
    numbers = []
    
    for file in existing_files:
        try:
            number = int(file.split('_')[-1].split('.')[0])
            numbers.append(number)
        except ValueError:
            continue
    
    return max(numbers) + 1 if numbers else 1

def process_csv(file_path):
    # Load the CSV file with flexible column names
    df = pd.read_csv(file_path, delimiter=';', skiprows=5, low_memory=False)
    df['DATE/TIME READ'] = pd.to_datetime(df['DATE/TIME READ'])
    df = df.sort_values(by='DATE/TIME READ')

    # Detect the columns present in the dataset
    rain_col = 'RAINFALL AMOUNT (mm)' if 'RAINFALL AMOUNT (mm)' in df.columns else None
    water_col = 'WATERLEVEL (m)' if 'WATERLEVEL (m)' in df.columns else None
    water_msl_col = 'WATERLEVEL MSL (m)' if 'WATERLEVEL MSL (m)' in df.columns else None
    date_column = 'DATE/TIME READ'

    # Initialize dataframes for anomalies
    relevant_data = pd.DataFrame()
    anomalies = pd.DataFrame()

    if rain_col:
        # Handle rainfall data
        irrelevant_data_criteria = (df[rain_col] == 0) & (df[date_column].diff().dt.total_seconds() > 3600)
        irrelevant_data = df[irrelevant_data_criteria]
        relevant_data = df[~irrelevant_data_criteria].copy()
        relevant_data['time_diff'] = relevant_data[date_column].diff().dt.total_seconds()
        relevant_data['anomaly'] = False

        for i in range(1, len(relevant_data)):
            current_value = relevant_data[rain_col].iloc[i]
            previous_value = relevant_data[rain_col].iloc[i-1]
            time_diff = relevant_data['time_diff'].iloc[i]

            if current_value == 0.0:
                continue
            elif (current_value != previous_value) and (time_diff < 1800):
                relevant_data.loc[i, 'anomaly'] = True
            elif (current_value == previous_value) and (time_diff >= 1800):
                continue
            else:
                relevant_data.loc[i, 'anomaly'] = True

        anomalies = relevant_data[relevant_data['anomaly']]
    elif water_col:
        # Handle water level data
        relevant_data = df.copy()
        relevant_data['time_diff'] = relevant_data[date_column].diff().dt.total_seconds()
        relevant_data['anomaly'] = False

        # Define a threshold for sudden changes
        change_threshold = 2.0  # Example threshold, adjust as needed

        for i in range(1, len(relevant_data)):
            current_value = relevant_data[water_col].iloc[i]
            previous_value = relevant_data[water_col].iloc[i-1]
            time_diff = relevant_data['time_diff'].iloc[i]

            if time_diff > 3600:
                continue

            if abs(current_value - previous_value) > change_threshold:
                relevant_data.loc[i, 'anomaly'] = True

        anomalies = relevant_data[relevant_data['anomaly']]
    else:
        irrelevant_data = pd.DataFrame()
        relevant_data = df.copy()
        anomalies = pd.DataFrame()

    return relevant_data, irrelevant_data, anomalies

def train_models(df):
    # Prepare features and labels based on available columns
    if 'RAINFALL AMOUNT (mm)' in df.columns:
        X = df[['RAINFALL AMOUNT (mm)', 'time_diff']]
    elif 'WATERLEVEL (m)' in df.columns:
        X = df[['WATERLEVEL (m)', 'time_diff']]
    else:
        X = pd.DataFrame()  # Handle cases with no relevant columns
    y = df['anomaly'].astype(int)

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    rf_y_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_y_pred)

    # Train Logistic Regression model
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    lr_y_pred = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_y_pred)

    return rf_model, lr_model, rf_accuracy, lr_accuracy

def save_dataframe(df, folder_path, base_name):
    file_number = get_next_file_number(folder_path, base_name)
    file_name = os.path.join(folder_path, f'{base_name}{file_number}.csv')
    df.to_csv(file_name, index=False)
    return file_name

def import_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        global relevant_data, irrelevant_data, anomalies, rf_model, lr_model, rf_accuracy, lr_accuracy
        relevant_data, irrelevant_data, anomalies = process_csv(file_path)
        if not relevant_data.empty:
            rf_model, lr_model, rf_accuracy, lr_accuracy = train_models(relevant_data)
            messagebox.showinfo("Success", 
                                f"CSV file processed successfully.\n"
                                f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%\n"
                                f"Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")
            show_download_buttons()
        else:
            messagebox.showwarning("No Data", "No relevant data found in the selected file.")

def import_csv_for_training():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        global relevant_data, rf_model, lr_model, rf_accuracy, lr_accuracy

        relevant_data, _, _ = process_csv(file_path)

        if not relevant_data.empty:
            rf_model, lr_model, rf_accuracy, lr_accuracy = train_models(relevant_data)
            messagebox.showinfo("Success", 
                                f"Model trained with the selected CSV file.\n"
                                f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%\n"
                                f"Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")
        else:
            messagebox.showwarning("No Data", "No relevant data found in the selected file.")

def download_relevant_data():
    folder_path = filedialog.askdirectory()
    if folder_path and relevant_data is not None:
        file_name = save_dataframe(relevant_data, folder_path, 'relevant_')
        messagebox.showinfo("Saved", f"Relevant data saved as {file_name}")

def download_irrelevant_data():
    folder_path = filedialog.askdirectory()
    if folder_path and irrelevant_data is not None:
        file_name = save_dataframe(irrelevant_data, folder_path, 'irrelevant_')
        messagebox.showinfo("Saved", f"Irrelevant data saved as {file_name}")

def download_anomalies():
    folder_path = filedialog.askdirectory()
    if folder_path and anomalies is not None:
        file_name = save_dataframe(anomalies, folder_path, 'anomalies_')
        messagebox.showinfo("Saved", f"Anomalies saved as {file_name}")

def show_download_buttons():
    download_label = Label(root, text="Download Processed Data:")
    download_label.pack(pady=10)

    download_relevant_btn = Button(root, text="Download Relevant Data", command=download_relevant_data)
    download_relevant_btn.pack(pady=5)

    download_irrelevant_btn = Button(root, text="Download Irrelevant Data", command=download_irrelevant_data)
    download_irrelevant_btn.pack(pady=5)

    download_anomalies_btn = Button(root, text="Download Anomalies", command=download_anomalies)
    download_anomalies_btn.pack(pady=5)

    if rf_accuracy is not None and lr_accuracy is not None:
        accuracy_label = Label(root, text=f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%\n"
                                          f"Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")
        accuracy_label.pack(pady=10)

def train_model_again():
    global rf_model, lr_model, rf_accuracy, lr_accuracy
    if relevant_data is not None:
        rf_model, lr_model, rf_accuracy, lr_accuracy = train_models(relevant_data)
        messagebox.showinfo("Training Complete", 
                            f"Model retrained successfully.\n"
                            f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%\n"
                            f"Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")
    else:
                messagebox.showwarning("No Data", "No data available to retrain the model.")

# Initialize global variables
relevant_data = irrelevant_data = anomalies = rf_model = lr_model = None
rf_accuracy = lr_accuracy = None

# Initialize Tkinter window
root = Tk()
root.title("CSV Data Processor")

Label(root, text="Import a CSV file to process:").pack(pady=10)
Button(root, text="Import CSV", command=import_csv).pack(pady=5)
Button(root, text="Import CSV for Training", command=import_csv_for_training).pack(pady=5)
Button(root, text="Retrain Model", command=train_model_again).pack(pady=5)

root.mainloop()


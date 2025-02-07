import pandas as pd
import requests

# Load the dataset
file_path = "Hackathon_Round_2.xlsx"
df = pd.read_excel(file_path)

# Keep only necessary columns & limit to 10 rows for testing
df = df[["Text", "subtask_a", "subtask_b", "subtask_c", "subtask_d"]].dropna().head(10)

# API URL
api_url = "http://127.0.0.1:8000/predict"

# Track accuracy
correct_a, correct_b, correct_c, correct_d = 0, 0, 0, 0

for index, row in df.iterrows():
    tweet = row["Text"]
    
    # Send request to API
    response = requests.post(api_url, json={"tweet": tweet})
    
    if response.status_code == 200:
        prediction = response.json()
        
        # Log API response vs actual labels
        print(f"\n🔹 **Tweet {index + 1}:** {tweet}")
        print(f"👉 **Actual Labels** | A: {row['subtask_a']} | B: {row['subtask_b']} | C: {row['subtask_c']} | D: {row['subtask_d']}")
        print(f"👉 **API Prediction** | A: {prediction['category']} | B: {prediction['emotion_category']} | C: {prediction['aspect_category']} | D: {prediction['subtask_d']}\n")

        # Compare with actual labels
        if prediction["category"] == row["subtask_a"]:
            correct_a += 1
        if prediction["emotion_category"] == row["subtask_b"]:
            correct_b += 1
        if "aspect_category" in prediction and prediction["aspect_category"] == row["subtask_c"]:
            correct_c += 1
        if "subtask_d" in prediction and prediction["subtask_d"] == row["subtask_d"]:
            correct_d += 1

# Calculate accuracy
accuracy_a = correct_a / len(df) * 100
accuracy_b = correct_b / len(df) * 100
accuracy_c = correct_c / len(df) * 100
accuracy_d = correct_d / len(df) * 100

# Print results
print(f"\n✅ Accuracy for Subtask A (COVID vs Non-COVID): {accuracy_a:.2f}%")
print(f"✅ Accuracy for Subtask B (Emotional vs Factual): {accuracy_b:.2f}%")
print(f"✅ Accuracy for Subtask C (Aspect Classification): {accuracy_c:.2f}%")
print(f"✅ Accuracy for Subtask D: {accuracy_d:.2f}%")

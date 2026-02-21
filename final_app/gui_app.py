import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import pickle
import os

# Load Model Safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Main Window
root = tk.Tk()
root.title("Train Journey Time Prediction System")
root.geometry("800x600")
root.configure(bg="#F4F6F9")

# Screen Switch Function

def show_prediction_screen():
    welcome_frame.pack_forget()
    prediction_frame.pack(fill="both", expand=True)

def go_back():
    prediction_frame.pack_forget()
    welcome_frame.pack(fill="both", expand=True)

def exit_program():
    root.destroy()


# Welcome Screen

welcome_frame = tk.Frame(root, bg="#F4F6F9")
welcome_frame.pack(fill="both", expand=True)

image_path = os.path.join(BASE_DIR, "train_image.png")
img = Image.open(image_path)
img = img.resize((450, 280))
photo = ImageTk.PhotoImage(img)

img_label = tk.Label(welcome_frame, image=photo, bg="#F4F6F9")
img_label.pack(pady=20)

title_label = tk.Label(
    welcome_frame,
    text="Welcome to Train Journey Time Prediction System",
    font=("Segoe UI", 20, "bold"),
    bg="#F4F6F9"
)
title_label.pack(pady=10)

start_button = tk.Button(
    welcome_frame,
    text="Start Prediction",
    font=("Segoe UI", 14, "bold"),
    bg="#007ACC",
    fg="white",
    width=20,
    height=2,
    command=show_prediction_screen
)
start_button.pack(pady=20)

exit_button = tk.Button(
    welcome_frame,
    text="End Program",
    font=("Segoe UI", 12),
    bg="#DC3545",
    fg="white",
    width=15,
    command=exit_program
)
exit_button.pack(pady=10)

# Prediction Screen

prediction_frame = tk.Frame(root, bg="#F4F6F9")

heading = tk.Label(
    prediction_frame,
    text="Enter Journey Details",
    font=("Segoe UI", 18, "bold"),
    bg="#F4F6F9"
)
heading.pack(pady=20)

# Distance
tk.Label(
    prediction_frame,
    text="Total Distance (km):",
    font=("Segoe UI", 13),
    bg="#F4F6F9"
).pack(pady=5)

distance_entry = tk.Entry(
    prediction_frame,
    font=("Segoe UI", 14),
    width=25,
    bd=3,
    relief="groove"
)
distance_entry.pack(pady=10)

# Stops
tk.Label(
    prediction_frame,
    text="Total Stops:",
    font=("Segoe UI", 13),
    bg="#F4F6F9"
).pack(pady=5)

stops_entry = tk.Entry(
    prediction_frame,
    font=("Segoe UI", 14),
    width=25,
    bd=3,
    relief="groove"
)
stops_entry.pack(pady=10)

# Progress Bar (Loading Animation)
progress = ttk.Progressbar(
    prediction_frame,
    orient="horizontal",
    length=300,
    mode="indeterminate"
)

# Result Label
result_label = tk.Label(
    prediction_frame,
    text="",
    font=("Segoe UI", 15, "bold"),
    fg="green",
    bg="#F4F6F9"
)
result_label.pack(pady=20)

# Prediction Function with Loading

def predict_duration():
    try:
        distance = float(distance_entry.get())
        stops = int(stops_entry.get())

        progress.pack(pady=10)
        progress.start(10)

        root.after(1500, lambda: complete_prediction(distance, stops))

    except:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")

def complete_prediction(distance, stops):
    progress.stop()
    progress.pack_forget()

    prediction = model.predict([[distance, stops]])
    total_minutes = prediction[0]

    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)

    result_label.config(
        text=f"Estimated Journey Duration: {hours} hours {minutes} minutes"
    )

# Predict Button
predict_button = tk.Button(
    prediction_frame,
    text="Predict Duration",
    font=("Segoe UI", 14, "bold"),
    bg="#28A745",
    fg="white",
    width=20,
    height=2,
    command=predict_duration
)
predict_button.pack(pady=20)

# Back Button
back_button = tk.Button(
    prediction_frame,
    text="Back",
    font=("Segoe UI", 12),
    bg="#6C757D",
    fg="white",
    width=15,
    command=go_back
)
back_button.pack(pady=5)

# End Button
end_button = tk.Button(
    prediction_frame,
    text="End Program",
    font=("Segoe UI", 12),
    bg="#DC3545",
    fg="white",
    width=15,
    command=exit_program
)
end_button.pack(pady=10)

root.mainloop()
# Spam Email Detection

A Machine Learning based web application that detects whether a message is **Spam** or **Not Spam** using Python, Flask, and Scikit-learn.

## Features

- Detects spam emails/messages
- User-friendly web interface
- Machine Learning model trained on spam datasets
- Fast and accurate predictions
- Built with Flask

## Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML
- CSS

## Project Structure

```
Spam-Email-Detection/
│
├── app.py
├── spam_detector.py
├── best_spam_detector.pkl
├── data/
├── templates/
│   └── index.html
└── README.md
```

## Dataset

The model is trained using spam message datasets containing labeled spam and ham (non-spam) messages.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/SantoshKumar436/Spam-Email-Detection.git
```

2. Navigate to the project directory:

```bash
cd Spam-Email-Detection
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

## Usage

1. Enter an email or message.
2. Click the Predict button.
3. The application will classify the message as:
   - Spam
   - Not Spam

## Future Improvements

- Email attachment analysis
- Deep Learning models
- Real-time email integration
- Enhanced UI/UX

## Author
** Santosh Kumar **

**Santosh Kumar**

GitHub: https://github.com/SantoshKumar436

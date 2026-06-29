import os
import re
import string
import zipfile
import tarfile
import requests
import joblib
import numpy as np
import pandas as pd
import email
from email import policy
from scipy.sparse import hstack

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# 1. Dataset Downloading and Extraction
# ==========================================

def download_file(url, dest_path):
    """Downloads a file using the requests library with user-agent headers."""
    print(f"Downloading: {url}")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers, stream=True, timeout=10)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Saved to: {dest_path}")

def extract_tar_bz2(filepath, extract_dir):
    """Extracts a .tar.bz2 file to the specified directory."""
    print(f"Extracting tar.bz2: {filepath}")
    with tarfile.open(filepath, "r:bz2") as tar:
        tar.extractall(path=extract_dir)
    print("Extraction completed.")

def extract_zip(filepath, extract_dir):
    """Extracts a .zip file to the specified directory."""
    print(f"Extracting zip: {filepath}")
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction completed.")

def load_spamassassin(data_dir):
    """Downloads and loads the SpamAssassin public corpus (Ham & Spam)."""
    os.makedirs(data_dir, exist_ok=True)
    
    # Primary and Alternate URLs for easy_ham and spam
    ham_urls = [
        "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
        "https://spamassassin.apache.org/publiccorpus/20030228_easy_ham.tar.bz2"
    ]
    spam_urls = [
        "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2",
        "https://spamassassin.apache.org/publiccorpus/20030228_spam.tar.bz2"
    ]
    
    ham_tar = os.path.join(data_dir, "easy_ham.tar.bz2")
    spam_tar = os.path.join(data_dir, "spam.tar.bz2")
    
    # Download Ham
    if not os.path.exists(ham_tar):
        downloaded = False
        for url in ham_urls:
            try:
                download_file(url, ham_tar)
                downloaded = True
                break
            except Exception as e:
                print(f"Warning: Failed to download Ham from {url} ({e})")
        if not downloaded:
            raise RuntimeError("Failed to download Easy Ham from all available URLs.")
            
    # Download Spam
    if not os.path.exists(spam_tar):
        downloaded = False
        for url in spam_urls:
            try:
                download_file(url, spam_tar)
                downloaded = True
                break
            except Exception as e:
                print(f"Warning: Failed to download Spam from {url} ({e})")
        if not downloaded:
            raise RuntimeError("Failed to download Spam from all available URLs.")

    # Extracting
    ham_extract_dir = os.path.join(data_dir, "easy_ham_extracted")
    spam_extract_dir = os.path.join(data_dir, "spam_extracted")
    
    if not os.path.exists(ham_extract_dir):
        extract_tar_bz2(ham_tar, ham_extract_dir)
    if not os.path.exists(spam_extract_dir):
        extract_tar_bz2(spam_tar, spam_extract_dir)

    # Locate and read easy_ham files
    emails = []
    labels = []  # 0: Ham, 1: Spam
    
    ham_folder = os.path.join(ham_extract_dir, "easy_ham")
    if not os.path.exists(ham_folder):
        # Find directory containing email files starting with '0' or having 'cmds'
        for root, dirs, files in os.walk(ham_extract_dir):
            if any(f.startswith("0") for f in files):
                ham_folder = root
                break
                
    for filename in os.listdir(ham_folder):
        if filename == "cmds" or filename.startswith("."):
            continue
        filepath = os.path.join(ham_folder, filename)
        if os.path.isdir(filepath):
            continue
        try:
            with open(filepath, 'rb') as f:
                emails.append(f.read())
                labels.append(0)
        except Exception as e:
            print(f"Error reading Ham file {filepath}: {e}")

    # Locate and read spam files
    spam_folder = os.path.join(spam_extract_dir, "spam")
    if not os.path.exists(spam_folder):
        for root, dirs, files in os.walk(spam_extract_dir):
            if any(f.startswith("0") for f in files):
                spam_folder = root
                break
                
    for filename in os.listdir(spam_folder):
        if filename == "cmds" or filename.startswith("."):
            continue
        filepath = os.path.join(spam_folder, filename)
        if os.path.isdir(filepath):
            continue
        try:
            with open(filepath, 'rb') as f:
                emails.append(f.read())
                labels.append(1)
        except Exception as e:
            print(f"Error reading Spam file {filepath}: {e}")
            
    return emails, labels

def load_uci_sms(data_dir):
    """Downloads and loads the UCI SMS Spam Collection dataset as a fallback."""
    os.makedirs(data_dir, exist_ok=True)
    zip_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    zip_path = os.path.join(data_dir, "smsspamcollection.zip")
    
    if not os.path.exists(zip_path):
        try:
            download_file(zip_url, zip_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download UCI SMS Spam Collection: {e}")
            
    extract_dir = os.path.join(data_dir, "uci_sms_extracted")
    if not os.path.exists(extract_dir):
        extract_zip(zip_path, extract_dir)
        
    tsv_path = os.path.join(extract_dir, "SMSSpamCollection")
    emails = []
    labels = []
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                label_str, text = parts
                label = 1 if label_str == 'spam' else 0
                # Construct mock email wrapper so downstream parses work identically
                mock_email = f"From: unknown@domain.com\nSubject: SMS Message\n\n{text}"
                emails.append(mock_email.encode('utf-8'))
                labels.append(label)
                
    return emails, labels

def load_dataset(data_dir="data"):
    """Loads dataset using the UCI SMS Spam Collection directly."""
    print("--- Loading UCI SMS Spam Collection directly ---")
    emails, labels = load_uci_sms(data_dir)
    print(f"Successfully loaded UCI SMS Spam Collection dataset: {len(emails)} messages.")
    return emails, labels


# ==========================================
# 2. Email Parsing and Text Preprocessing
# ==========================================

def safe_decode(payload, charset):
    """Safely decodes raw bytes into a string, falling back to utf-8/latin-1 if charset is unrecognized."""
    if not payload:
        return ""
    if not charset:
        charset = 'utf-8'
    try:
        return payload.decode(charset, errors='ignore')
    except (LookupError, ValueError, UnicodeDecodeError):
        try:
            return payload.decode('utf-8', errors='ignore')
        except Exception:
            return payload.decode('latin-1', errors='ignore')

def parse_raw_email(raw_content):
    """Parses raw email bytes or string into Subject, Sender (From), and Body using the email library."""
    if isinstance(raw_content, bytes):
        msg = email.message_from_bytes(raw_content, policy=policy.default)
    else:
        msg = email.message_from_string(raw_content, policy=policy.default)
        
    subject = msg.get('Subject', '') or ''
    sender = msg.get('From', '') or ''
    
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            if content_type in ["text/plain", "text/html"] and "attachment" not in content_disposition:
                payload = part.get_payload(decode=True)
                if payload:
                    body += safe_decode(payload, part.get_content_charset())
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body = safe_decode(payload, msg.get_content_charset())
        else:
            body = msg.get_payload() or ""
            
    return {
        'subject': str(subject),
        'sender': str(sender),
        'body': str(body)
    }

def preprocess_text(text):
    """Cleans text: lowercase, remove HTML tags, strip punctuation, remove stop words, tokenize."""
    if not text:
        return ""
    # 1. Lowercase
    text = text.lower()
    # 2. Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # 3. Strip punctuation (replace with spaces to avoid joining words)
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    # 4. Tokenize
    tokens = text.split()
    # 5. Remove stop words
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
    # Rejoin into preprocessed string
    return " ".join(tokens)


# ==========================================
# 3. Feature Extraction (Metadata Features)
# ==========================================

def extract_meta_features(parsed_emails):
    """
    Extracts custom metadata features from raw parsed emails:
    - Caps ratio in subject
    - URL count
    - Exclamation marks count
    - URL shortener presence
    - Free email domain flag
    - Numeric username ratio
    - Sender domain suspicion score
    """
    meta_features = []
    
    # Common URL shorteners
    shorteners = {'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'rebrand.ly', 'is.gd', 'buff.ly', 'ow.ly', 'bit.do', 'adf.ly'}
    # Common free email domains
    free_domains = {'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com', 'mail.com', 'zoho.com', 'yandex.com', 'protonmail.com', 'gmx.com', 'icloud.com'}
    # Suspicious TLDs
    suspicious_tlds = {'zip', 'mov', 'xyz', 'biz', 'info', 'top', 'click', 'download', 'win', 'party', 'date', 'science', 'gq', 'cf', 'tk', 'ml', 'ga', 'club', 'work'}

    for email_dict in parsed_emails:
        subject = email_dict.get('subject', '') or ''
        sender = email_dict.get('sender', '') or ''
        body = email_dict.get('body', '') or ''
        
        # 1. Caps ratio in subject
        alpha_chars = sum(1 for c in subject if c.isalpha())
        caps_ratio = sum(1 for c in subject if c.isupper()) / alpha_chars if alpha_chars > 0 else 0.0
        
        # 2. URL count in body
        urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', body)
        url_count = len(urls)
        
        # 3. Exclamation marks count in subject + body
        excl_count = body.count('!') + subject.count('!')
        
        # 4. Has URL shortener
        has_shortener = 1.0 if any(any(s in u.lower() for s in shorteners) for u in urls) else 0.0
        
        # Parse email components
        domain = ""
        username = ""
        domain_match = re.search(r'([\w.-]+)@([\w.-]+)', sender)
        if domain_match:
            username = domain_match.group(1)
            domain = domain_match.group(2).lower()
        else:
            domain = sender.lower()
            username = sender
            
        # 5. Is free domain
        is_free_domain = 1.0 if domain in free_domains else 0.0
        
        # 6. Numeric username ratio
        username_len = len(username)
        numeric_username_ratio = sum(c.isdigit() for c in username) / username_len if username_len > 0 else 0.0
        
        # 7. Sender domain suspicion score
        suspicion_score = 0.0
        if not domain:
            suspicion_score += 1.0
        else:
            parts = domain.split('.')
            tld = parts[-1] if len(parts) > 1 else ""
            if tld in suspicious_tlds:
                suspicion_score += 1.0
            if any(c.isdigit() for c in domain):
                suspicion_score += 0.5
            if domain.count('-') > 1:
                suspicion_score += 0.5
                
        meta_features.append([
            float(caps_ratio),
            float(url_count),
            float(excl_count),
            float(suspicion_score),
            float(has_shortener),
            float(is_free_domain),
            float(numeric_username_ratio)
        ])
        
    return np.array(meta_features, dtype=float)


# ==========================================
# 4. Wrapper Class for Model Persistence
# ==========================================

class SpamDetector:
    """Wrapper class encapsulating preprocessing, feature scaling, classification, and scoring."""
    def __init__(self, feature_method, classifier, vectorizer, scaler=None):
        self.feature_method = feature_method  # 'bow', 'tfidf', or 'combined'
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.scaler = scaler  # MinMaxScaler instance (used if 'combined')
        
    def predict(self, raw_email_string):
        """Classifies raw email text as 'spam' or 'ham'."""
        email_dict = parse_raw_email(raw_email_string)
        text = email_dict['body'] + " " + email_dict['subject']
        clean_text = preprocess_text(text)
        
        X_text = self.vectorizer.transform([clean_text])
        
        if self.feature_method == 'combined':
            meta = extract_meta_features([email_dict])
            meta_scaled = self.scaler.transform(meta)
            X_features = hstack([X_text, meta_scaled]).tocsr()
        else:
            X_features = X_text
            
        pred = self.classifier.predict(X_features)[0]
        return "spam" if pred == 1 else "ham"
        
    def predict_proba(self, raw_email_string):
        """Returns the confidence score (ham_prob, spam_prob) for the classification."""
        email_dict = parse_raw_email(raw_email_string)
        text = email_dict['body'] + " " + email_dict['subject']
        clean_text = preprocess_text(text)
        
        X_text = self.vectorizer.transform([clean_text])
        
        if self.feature_method == 'combined':
            meta = extract_meta_features([email_dict])
            meta_scaled = self.scaler.transform(meta)
            X_features = hstack([X_text, meta_scaled]).tocsr()
        else:
            X_features = X_text
            
        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(X_features)[0]
            return float(probs[0]), float(probs[1])
        elif hasattr(self.classifier, "decision_function"):
            # Calibrate SVM output using sigmoid
            decision = self.classifier.decision_function(X_features)[0]
            prob_spam = 1 / (1 + np.exp(-decision))
            return 1.0 - float(prob_spam), float(prob_spam)
        else:
            # Fallback (binary indicator)
            pred = self.classifier.predict(X_features)[0]
            return (0.0, 1.0) if pred == 1 else (1.0, 0.0)


# ==========================================
# 5. Main Model Comparison and Training Flow
# ==========================================

def main():
    # Load dataset (SpamAssassin with UCI SMS fallback)
    raw_emails, labels = load_dataset("data")
    labels = np.array(labels)
    
    # Parse all emails
    print("\nParsing raw emails...")
    parsed_emails = []
    for raw in raw_emails:
        parsed_emails.append(parse_raw_email(raw))
    print("Parsing completed.")
    
    # Preprocess texts (combining body and subject)
    print("Preprocessing texts...")
    cleaned_texts = [preprocess_text(e['body'] + " " + e['subject']) for e in parsed_emails]
    print("Preprocessing completed.")
    
    # Perform 80/20 train/test split on parsed structures
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        parsed_emails, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Prepare text blocks for vectorization
    train_texts = [preprocess_text(e['body'] + " " + e['subject']) for e in X_train_raw]
    test_texts = [preprocess_text(e['body'] + " " + e['subject']) for e in X_test_raw]
    
    # ------------------------------------------
    # Feature Extraction Setup
    # ------------------------------------------
    print("\n--- Fitting Vectorizers and Scalers ---")
    
    # Method 1: Bag-of-Words
    bow_vectorizer = CountVectorizer(lowercase=False)
    X_train_bow = bow_vectorizer.fit_transform(train_texts)
    X_test_bow = bow_vectorizer.transform(test_texts)
    
    # Method 2: TF-IDF
    tfidf_vectorizer = TfidfVectorizer(lowercase=False)
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
    X_test_tfidf = tfidf_vectorizer.transform(test_texts)
    
    # Method 3: Combined (TF-IDF + Metadata)
    print("Extracting meta features...")
    X_train_meta = extract_meta_features(X_train_raw)
    X_test_meta = extract_meta_features(X_test_raw)
    
    # MinMaxScaler preserves non-negative range [0, 1] for MultinomialNB
    scaler = MinMaxScaler()
    X_train_meta_scaled = scaler.fit_transform(X_train_meta)
    X_test_meta_scaled = scaler.transform(X_test_meta)
    
    X_train_combined = hstack([X_train_tfidf, X_train_meta_scaled]).tocsr()
    X_test_combined = hstack([X_test_tfidf, X_test_meta_scaled]).tocsr()
    
    # Setup models
    models = {
        'Multinomial NB': MultinomialNB(),
        'Linear SVM': LinearSVC(random_state=42, max_iter=3000),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    feature_sets = {
        'Bag-of-Words': (X_train_bow, X_test_bow, bow_vectorizer, None),
        'TF-IDF': (X_train_tfidf, X_test_tfidf, tfidf_vectorizer, None),
        'Combined (TF-IDF + Meta)': (X_train_combined, X_test_combined, tfidf_vectorizer, scaler)
    }
    
    results = []
    best_f1 = -1.0
    best_model_obj = None
    best_model_name = ""
    best_feature_name = ""
    
    # Evaluate combinations
    for feat_name, (X_tr, X_te, vec, scl) in feature_sets.items():
        print(f"\nEvaluating models on {feat_name}...")
        for model_name, model in models.items():
            # Clone model to prevent reuse/side-effects
            from sklearn.base import clone
            clf = clone(model)
            
            # Train
            clf.fit(X_tr, y_train)
            
            # Predict
            y_pred = clf.predict(X_te)
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            
            results.append({
                'Feature Method': feat_name,
                'Model': model_name,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1,
                'Confusion Matrix': cm
            })
            
            # Print individual model reports
            print(f"  > {model_name}: F1={f1:.4f} | Acc={acc:.4f}")
            print(f"    Confusion Matrix:\n    {cm[0]}\n    {cm[1]}")
            
            # Select best model based on F1-score
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = model_name
                best_feature_name = feat_name
                
                # Determine configuration name for the wrapper
                method_key = 'bow' if feat_name == 'Bag-of-Words' else ('tfidf' if feat_name == 'TF-IDF' else 'combined')
                best_model_obj = SpamDetector(method_key, clf, vec, scl)
                
    # ------------------------------------------
    # Comparison Summary Table
    # ------------------------------------------
    print("\n" + "="*80)
    print("                           SUMMARY EVALUATION TABLE")
    print("="*80)
    df_results = pd.DataFrame(results)
    # Format floating numbers for readable output
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    print(df_results[['Feature Method', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].to_string(index=False))
    print("="*80)
    
    # ------------------------------------------
    # Save the Best Model
    # ------------------------------------------
    print(f"\nBest Model identified: {best_model_name} using {best_feature_name} (F1-Score: {best_f1:.4f})")
    model_filename = "best_spam_detector.pkl"
    joblib.dump(
    {
        "feature_method": best_model_obj.feature_method,
        "classifier": best_model_obj.classifier,
        "vectorizer": best_model_obj.vectorizer,
        "scaler": best_model_obj.scaler,
    },
    model_filename,
)
    print(f"Saved the best model wrapper to: {os.path.abspath(model_filename)}")
    
    # ------------------------------------------
    # Inference Extensions Demonstration
    # ------------------------------------------
    print("\n" + "="*80)
    print("                    INFERENCE MODEL EXTENSION DEMONSTRATION")
    print("="*80)
    print("Reloading model from disk...")
    loaded_detector = joblib.load(model_filename)
    
    test_emails = [
        # Example 1: Clear Spam
        """Subject: $$$ EXCLUSIVE LOAN OFFER $$$
From: contact@fast-cash-now.click
Date: Mon, 15 Jun 2026

Dear customer,
You qualify for a pre-approved, no-credit-check loan of $10,000! 
Hurry up and click here http://bit.ly/cashnow123 to claim it immediately!
Guaranteed approval!!!
""",
        # Example 2: Clear Ham (Personal/Work)
        """Subject: Project Status Review Meeting
From: alice.smith@gmail.com
Date: Mon, 15 Jun 2026

Hi team,
Let's meet tomorrow at 10:00 AM to review the current status of the spam detection model project.
Please make sure your code changes are pushed to git beforehand.

Best,
Alice
""",
        # Example 3: Spam disguised with headers and links
        """Subject: Win a brand new smartphone today!
From: info@sweepstakes2026.xyz
Date: Mon, 15 Jun 2026

Congrats! Your email address was selected in our daily lottery!
Visit www.prize-winner.biz/phone to secure your prize now!
"""
    ]
    
    for i, raw_email in enumerate(test_emails, 1):
        parsed = parse_raw_email(raw_email)
        label = loaded_detector.predict(raw_email)
        ham_prob, spam_prob = loaded_detector.predict_proba(raw_email)
        
        print(f"\n[Test Email #{i}]")
        print(f"Sender:  {parsed['sender']}")
        print(f"Subject: {parsed['subject']}")
        print(f"Result:  {label.upper()} (Confidence score: {spam_prob:.4f})")
        print("-" * 40)
        
if __name__ == "__main__":
    main()

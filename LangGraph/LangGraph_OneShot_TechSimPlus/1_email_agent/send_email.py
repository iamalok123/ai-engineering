import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
load_dotenv()
import os



# 1. Define configuration variables

def send_mail(recipient_email: str, subject: str, body: str) -> str:    
    if not recipient_email or not recipient_email.strip():
        return "Failed to send email: Recipient email address is missing."
        
    # Define email details
    smtp_server = "smtp.gmail.com"
    smtp_port = 587  
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("PASSWORD")  

    if not sender_email or not sender_password:
        return "Failed to send email: SENDER_EMAIL or PASSWORD not configured in environment variables."

    # Create the message
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient_email

    # Connect to the SMTP server and send
    try:
        # Use port 465 for SMTP_SSL, or port 587 for STARTTLS
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.ehlo()
            server.starttls()  # Upgrade the insecure connection to TLS
            server.ehlo()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return "Email sent successfully!"
    except Exception as e:
        return f"Failed to send email: {e}"
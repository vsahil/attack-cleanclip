import subprocess
import time
import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

SLACK_BOT_TOKEN = 'xoxb-14627938628-5735097798368-TiMXAxpzM9ubHYTtaJvglBrO'
CHANNEL_ID = 'C05LUERBXT7'

slack_client = WebClient(token=SLACK_BOT_TOKEN)


def get_gpu_memory_usage():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
        return list(map(int, result.decode('utf-8').strip().split('\n')))
    except Exception as e:
        print(f"Error in fetching GPU status: {e}")
        return []


def send_slack_message(message):
    # This function now returns the timestamp of the message, which we can use to track replies.
    try:
        response = slack_client.chat_postMessage(channel=CHANNEL_ID, text=message)
        return response["ts"]  # Timestamp of the message
    except SlackApiError as e:
        print(f"Error in sending slack message: {e}")
        return None


def listen_for_ok_reply(timestamp):
    # Check the thread replies for the given message timestamp and look for an "ok" reply.
    print(f"Listening for replies to message with timestamp: {timestamp}")
    try:
        for _ in range(30):     # Check for 5 minutes
            response = slack_client.conversations_replies(channel=CHANNEL_ID, ts=timestamp)
            messages = response["messages"]
            # print(f"Messages: {messages}")
            for message in messages:
                if message["text"].lower() == "ok" and float(message["ts"]) > float(timestamp):  # Avoid matching the original message
                    return True
            time.sleep(10)  # Check every 10 seconds
    except SlackApiError as e:
        print(f"Error in listening for replies: {e}")


def monitor_gpu_for_5mins():
    zero_memory_count = 0
    done = False
    while True:
        memory_usages = get_gpu_memory_usage()
        print(f"Memory usages: {memory_usages}", zero_memory_count)
        if memory_usages and memory_usages[0] < 100:
            zero_memory_count += 1  # Increment the count
        else:
            zero_memory_count = 0

        if zero_memory_count >= 7:      # if inactivity for 3.5 minutes
            done = True
            break

        time.sleep(30)  # Sleep for 30 seconds and check again. 
    if done:
        return True
    return False


def main():
    while True:
        # First check the GPU status for 5 minutes continuously
        if monitor_gpu_for_5mins():  # This function should return True if GPU is idle for 5 mins
            message_timestamp = send_slack_message("Alert: GPU 0 might have zero memory usage!")
            if listen_for_ok_reply(message_timestamp):  # Listen for an "ok" reply for 5 mins -- if we get a reply, sleep for 6 hours
                print("GPU 0 is idle for 5 mins and got an ok reply -- sleeping for 6 hours")
                time.sleep(6 * 60 * 60)  # Sleep for 6 hours once "ok" reply detected.
            else:
                print("GPU 0 is idle for 5 mins but no ok reply -- sleeping for 2 hours")
                time.sleep(2 * 60 * 60)  # If no "ok" reply detected, sleep for 2 hour, and then check the GPU again. 
        # else:
        #     print("GPU 0 is not idle for 5 mins -- sleeping for 1 hour and then checking again")
        #     time.sleep(1 * 60 * 60)  # If GPU was not idle for 5 mins, sleep for 1 hour and then check again. 

if __name__ == "__main__":
    main()

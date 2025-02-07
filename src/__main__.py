from core.video_processor import VideoProcessor

def main():
    print("Starting the video processor.")
    processor = VideoProcessor()
    processor.process_episodes()
    print("Finished the video processor.")

if __name__ == "__main__":
    main()
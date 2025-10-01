# from Core.main import main
import audio_app

if __name__ == "__main__":
    try:
        audio_app.main()
    except Exception as e:
        print(f"Cannot run executable: {e}")
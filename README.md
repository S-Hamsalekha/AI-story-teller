AI-story-teller

This project uses AI to convert short stories (given as text input) to  videos with audio narration and captions.





Control Flow of the system

1\.	Story Input

&nbsp;		User provides a story prompt (e.g., from Indian mythology).

2\.	Segmentation

&nbsp;		Gemini-2.5-Flash API splits the story into 5–7 scenes.

&nbsp;		Generates narration text and image prompts.

3\.	Image Generation

&nbsp;		Scene visuals generated using Imagen-3.0-generate-002 API.

4\.	Scene Processing

&nbsp;		Narration audio created via gTTS (English, Kannada) or ElevenLabs.

&nbsp;		Scene image animated using MoviePy (pan/zoom).

&nbsp;		Subtitles overlaid using PIL (Python Imaging Library).

5\.	Final Assembly

&nbsp;		All scenes combined into a complete narrated video using MoviePy.





==================================================================
Requirements
\[x]You will need a google colab / Kaggle account to use this
\[x]Api keys for Gemini , Imagen3.0 apis , 11labs apis



==================================================================



Usage

\[x] Upload the program onto a google colab or Kaggle notebook

\[x] Ensure all the required api keys are accessible from your environment

\[x] Run the program

\[x] Input your story when prompted

\[x] The final video will be available for download in 3-4 minutes.


"""Streamlit app: video/YouTube -> description + accurate timestamped chapters."""

import itertools
import traceback

import streamlit as st

import enhance
from acquire import AcquisitionError, acquire_from_upload, acquire_from_youtube
from generate import GenerationError, format_timestamp, generate
from transcribe import TranscriptionError, get_transcript

st.set_page_config(
    page_title="VidPrep",
    page_icon=":material/schedule:",
    layout="wide",
)

st.title(":material/schedule: VidPrep")
st.caption(
    "Upload a video file or paste a YouTube link. Timestamps are resolved from "
    "real transcribed word timings — never guessed by the model."
)

st.header("Input")

with st.container(border=True):
    input_mode = st.radio("Input", ["Upload video file", "YouTube link"], horizontal=True)

    uploaded_file = None
    youtube_url = None

    if input_mode == "Upload video file":
        uploaded_file = st.file_uploader(
            "Video or audio file", type=["mp4", "mov", "mkv", "webm", "ogv", "m4a", "mp3", "wav", "flac", "aac", "ogg"]
        )
    else:
        youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

    prefer_captions = st.toggle(
        "Prefer existing YouTube captions over AI_TRANSCRIBE when available",
        value=True,
        help="Faster and free, but coarser (cue-level) timing than AI_TRANSCRIBE's word-level output.",
    )
    ignore_existing_timestamps = st.toggle(
        "Ignore existing timestamps (always transcribe with AI_TRANSCRIBE)",
        value=False,
        help="Skip YouTube captions entirely and always re-transcribe with Snowflake AI_TRANSCRIBE's "
        "word-level timestamps, even when captions exist. Overrides the toggle above.",
    )
    use_captions_if_available = prefer_captions and not ignore_existing_timestamps

    generate_clicked = st.button("Generate", type="primary")

if generate_clicked:
    if input_mode == "Upload video file" and uploaded_file is None:
        st.warning("Please upload a file first.")
    elif input_mode == "YouTube link" and not youtube_url:
        st.warning("Please paste a YouTube URL first.")
    else:
        progress = st.empty()
        try:
            progress.progress(0, text="Acquiring media...")
            if input_mode == "Upload video file":
                media = acquire_from_upload(uploaded_file)
            else:
                media = acquire_from_youtube(youtube_url, prefer_captions=use_captions_if_available)
            progress.progress(25, text=f"Acquired: {media.title or 'media'}")

            progress.progress(40, text="Transcribing (this is the long step)...")
            transcript = get_transcript(media, use_captions_if_available=use_captions_if_available)
            progress.progress(
                80,
                text=f"Transcribed ({transcript.granularity}-level, {transcript.source}, "
                f"{len(transcript.words)} segments, {transcript.duration:.0f}s)",
            )

            progress.progress(90, text="Generating description and chapters...")
            result = generate(media, transcript)
            progress.progress(100, text="Done")
            progress.empty()

            st.session_state["media"] = media
            st.session_state["result"] = result
            st.session_state["transcript"] = transcript
            st.session_state["seek_seconds"] = 0.0
            st.session_state["extras"] = {}

        except AcquisitionError as e:
            progress.empty()
            st.error(f"Could not acquire media: {e}")
        except TranscriptionError as e:
            progress.empty()
            st.error(f"Transcription failed: {e}")
        except GenerationError as e:
            progress.empty()
            st.error(f"Generation failed: {e}")
        except Exception as e:
            progress.empty()
            st.error(f"Unexpected error: {e}")
            st.code(traceback.format_exc())

if "result" in st.session_state:
    media = st.session_state["media"]
    result = st.session_state["result"]
    transcript = st.session_state["transcript"]
    st.session_state.setdefault("seek_seconds", 0.0)
    st.session_state.setdefault("extras", {})
    extras = st.session_state["extras"]

    st.header("Output")

    with st.container(border=True):
        tab_overview, tab_titles_seo, tab_thumbs, tab_captions = st.tabs(
            ["Overview", "Titles & SEO", "Thumbnails & Clips", "Captions & FAQ"]
        )

        with tab_overview:
            left, right = st.columns([3, 2])

            # NOTE: this block must run before the "right" block below so that a
            # button click updates seek_seconds in session_state before the video
            # player (in the right column) reads it in this same script run.
            with left:
                st.subheader("Chapters")

                youtube_base_url = f"https://youtu.be/{media.video_id}" if media.video_id else None

                if youtube_base_url:
                    st.caption("Click a timestamp to open that moment on YouTube in a new tab.")

                    if media.existing_chapters:
                        hdr_cols = st.columns(2)
                        hdr_cols[0].markdown("**Generated chapters**")
                        hdr_cols[1].markdown("**Original YouTube chapters**")

                        for gen, orig in itertools.zip_longest(result.chapters, media.existing_chapters):
                            row_cols = st.columns(2)
                            with row_cols[0]:
                                if gen:
                                    ts = format_timestamp(gen.start_seconds)
                                    url = f"{youtube_base_url}?t={int(gen.start_seconds)}"
                                    st.markdown(
                                        f'<a href="{url}" target="_blank">{ts}</a> &nbsp;**{gen.title}**',
                                        unsafe_allow_html=True,
                                    )
                            with row_cols[1]:
                                if orig:
                                    st.write(f"{format_timestamp(orig.start_seconds)} {orig.title}")
                    else:
                        for c in result.chapters:
                            ts = format_timestamp(c.start_seconds)
                            url = f"{youtube_base_url}?t={int(c.start_seconds)}"
                            st.markdown(
                                f'<a href="{url}" target="_blank">{ts}</a> &nbsp;**{c.title}**',
                                unsafe_allow_html=True,
                            )
                else:
                    st.caption("Click a timestamp to jump the preview player to that moment.")

                    for i, c in enumerate(result.chapters):
                        ts = format_timestamp(c.start_seconds)
                        row_cols = st.columns([1, 4])
                        with row_cols[0]:
                            if st.button(ts, key=f"seek_btn_{i}"):
                                st.session_state["seek_seconds"] = c.start_seconds
                        with row_cols[1]:
                            st.markdown(f"**{c.title}**")

            with right:
                st.subheader("Preview")
                seek_seconds = int(st.session_state["seek_seconds"])
                if media.source_type == "youtube":
                    if media.local_preview_path:
                        st.video(media.local_preview_path, start_time=seek_seconds)
                    else:
                        st.caption(
                            "A video preview couldn't be downloaded for this video. Audio only:"
                        )
                        st.audio(media.local_media_path)
                elif media.media_kind == "video":
                    st.video(media.local_media_path, start_time=seek_seconds)
                else:
                    st.audio(media.local_media_path, start_time=seek_seconds)

            st.subheader("Description")
            lines = [result.description, "", "⏳ Timestamps"]
            for c in result.chapters:
                lines.append(f"{format_timestamp(c.start_seconds)} {c.title}")
            if result.keywords:
                lines += ["", "🔑 Keywords", ", ".join(result.keywords)]
            st.code("\n".join(lines), language=None)

        with tab_titles_seo:
            if st.button("Generate title & SEO suggestions", key="gen_titles_seo"):
                try:
                    with st.spinner("Generating title, category, and end-screen suggestions..."):
                        extras["titles_seo"] = enhance.generate_titles_seo(media, result, transcript.duration)
                except GenerationError as e:
                    st.error(f"Generation failed: {e}")

            titles_seo = extras.get("titles_seo")
            if titles_seo:
                st.subheader("Title suggestions")
                st.code("\n".join(titles_seo.titles), language=None)

                st.subheader("Category")
                st.write(f"**{titles_seo.category or 'Unknown'}** — {titles_seo.category_reason}")

                st.subheader("End screen")
                st.write(
                    f"Around **{format_timestamp(titles_seo.endscreen_seconds)}**: "
                    f"{titles_seo.endscreen_suggestion}"
                )

                st.subheader("SEO checklist")
                for item in enhance.seo_checklist(result, titles_seo.titles):
                    icon = ":material/check_circle:" if item["passed"] else ":material/warning:"
                    st.write(f"{icon} **{item['label']}** — {item['detail']}")
            else:
                st.caption("Click the button above to generate title, category, and SEO suggestions.")

        with tab_thumbs:
            st.subheader("Thumbnail candidates")
            if st.button("Generate thumbnail candidates", key="gen_thumbs"):
                extras["thumbnails"] = enhance.generate_thumbnails(media, result)

            thumbnails = extras.get("thumbnails")
            if thumbnails:
                thumb_cols = st.columns(min(len(thumbnails), 5) or 1)
                for i, t in enumerate(thumbnails):
                    with thumb_cols[i % len(thumb_cols)]:
                        st.image(t.image_path, caption=f"{format_timestamp(t.seconds)} — {t.chapter_title}")
            elif thumbnails == []:
                st.caption("No video available to extract thumbnail frames from (audio-only source).")
            else:
                st.caption("Click the button above to extract candidate thumbnail frames.")

            st.divider()
            st.subheader("Pull quotes / social clips")
            if st.button("Generate pull quotes", key="gen_quotes"):
                try:
                    with st.spinner("Scanning transcript for quotable moments..."):
                        extras["quotes"] = enhance.generate_quotes(media, transcript)
                except GenerationError as e:
                    st.error(f"Generation failed: {e}")

            quotes = extras.get("quotes")
            if quotes:
                for q in quotes:
                    st.markdown(f"> {q.text}")
                    st.caption(f"{format_timestamp(q.start_seconds)} - {format_timestamp(q.end_seconds)}")
            elif quotes == []:
                st.caption("No quotable moments found.")
            else:
                st.caption("Click the button above to find quotable moments for social/teaser clips.")

        with tab_captions:
            st.subheader("Caption file export")
            srt_text = enhance.build_srt(transcript)
            vtt_text = enhance.build_vtt(transcript)
            base_name = (media.title or media.video_id or "captions").replace(" ", "_")
            dl_cols = st.columns(2)
            with dl_cols[0]:
                st.download_button("Download .srt", srt_text, file_name=f"{base_name}.srt", mime="text/plain")
            with dl_cols[1]:
                st.download_button("Download .vtt", vtt_text, file_name=f"{base_name}.vtt", mime="text/vtt")

            st.divider()
            st.subheader("FAQ / Timestamps for X")
            if st.button("Generate FAQ", key="gen_faq"):
                try:
                    with st.spinner("Generating FAQ from chapters..."):
                        extras["faq"] = enhance.generate_faq(media, result)
                except GenerationError as e:
                    st.error(f"Generation failed: {e}")

            faq = extras.get("faq")
            if faq:
                lines = []
                for item in faq:
                    st.markdown(f"**{item.question}**")
                    st.write(item.answer)
                    st.caption(f"See {format_timestamp(item.start_seconds)} — {item.chapter_title}")
                    lines.append(f"{format_timestamp(item.start_seconds)} {item.question}")
                st.subheader("Copy-paste \"Timestamps for X\" block")
                st.code("\n".join(lines), language=None)
            elif faq == []:
                st.caption("No FAQ items generated.")
            else:
                st.caption("Click the button above to generate an FAQ from the chapters.")

import html
import json
from pathlib import Path

import streamlit.components.v1 as components

from src.signaflow_runtime import video_to_data_uri


def play_sign_sequence(
    videos,
    height=440,
):

    valid = []

    for item in videos:

        path = Path(
            item["path"]
        )

        if not path.exists():
            continue

        uri = video_to_data_uri(
            path
        )

        if uri:

            valid.append(
                {
                    "word": item["word"].upper(),
                    "src": uri,
                }
            )

    if not valid:

        components.html(
            """
            <div style="
                height:400px;
                display:flex;
                align-items:center;
                justify-content:center;
                background:#0b0e14;
                border-radius:20px;
                color:#737985;
                font-family:-apple-system,BlinkMacSystemFont,sans-serif;
                font-size:15px;
            ">
                No sign videos available for this sentence.
            </div>
            """,
            height=height,
        )

        return

    sequence_json = json.dumps(
        valid
    )

    page = f"""
    <!DOCTYPE html>

    <html>

    <head>

    <style>

        * {{
            box-sizing:border-box;
        }}

        html, body {{
            margin:0;
            padding:0;
            width:100%;
            height:100%;
            overflow:hidden;
            background:#080b11;
            font-family:
                -apple-system,
                BlinkMacSystemFont,
                "SF Pro Display",
                sans-serif;
        }}

        .container {{
            width:100%;
            height:100%;
            position:relative;
            background:
                radial-gradient(
                    circle at 50% 30%,
                    #171d29,
                    #080b11 70%
                );
            border-radius:20px;
            overflow:hidden;
        }}

        video {{
            width:100%;
            height:calc(100% - 64px);
            object-fit:contain;
            display:block;
            background:#080b11;
        }}

        .top {{
            position:absolute;
            top:14px;
            left:16px;
            right:16px;
            display:flex;
            justify-content:space-between;
            align-items:center;
            z-index:5;
            pointer-events:none;
        }}

        .label {{
            color:#8e95a3;
            font-size:10px;
            font-weight:700;
            letter-spacing:1.6px;
        }}

        .word {{
            color:white;
            font-size:15px;
            font-weight:700;
            letter-spacing:2px;
        }}

        .bottom {{
            position:absolute;
            left:0;
            right:0;
            bottom:0;
            height:64px;
            display:flex;
            align-items:center;
            padding:0 18px;
            background:rgba(8,11,17,.92);
            border-top:1px solid rgba(255,255,255,.07);
        }}

        .progress {{
            display:flex;
            gap:5px;
            flex:1;
        }}

        .dot {{
            height:4px;
            flex:1;
            border-radius:10px;
            background:#303642;
            transition:background .25s ease;
        }}

        .dot.active {{
            background:#ffffff;
        }}

        .count {{
            margin-left:14px;
            color:#707784;
            font-size:11px;
        }}

    </style>

    </head>

    <body>

    <div class="container">

        <div class="top">

            <div class="label">
                SIGN VISUALIZER
            </div>

            <div id="word"
                 class="word">
                —
            </div>

        </div>

        <video
            id="player"
            controls
            playsinline
            muted
            autoplay>
        </video>

        <div class="bottom">

            <div
                id="progress"
                class="progress">
            </div>

            <div
                id="count"
                class="count">
            </div>

        </div>

    </div>


    <script>

        const sequence =
            {sequence_json};

        const player =
            document.getElementById(
                "player"
            );

        const word =
            document.getElementById(
                "word"
            );

        const progress =
            document.getElementById(
                "progress"
            );

        const count =
            document.getElementById(
                "count"
            );

        let index = 0;


        function buildProgress() {{

            progress.innerHTML = "";

            sequence.forEach(
                (_, i) => {{

                    const dot =
                        document.createElement(
                            "div"
                        );

                    dot.className =
                        "dot";

                    dot.id =
                        "dot-" + i;

                    progress.appendChild(
                        dot
                    );
                }}
            );
        }}


        function updateProgress() {{

            sequence.forEach(
                (_, i) => {{

                    const dot =
                        document.getElementById(
                            "dot-" + i
                        );

                    if (!dot) return;

                    dot.classList.toggle(
                        "active",
                        i === index
                    );
                }}
            );

            count.textContent =
                `${{index + 1}} / ${{sequence.length}}`;
        }}


        function playCurrent() {{

            if (
                index >= sequence.length
            ) {{
                index = 0;
            }}

            const item =
                sequence[index];

            word.textContent =
                item.word;

            player.src =
                item.src;

            updateProgress();

            player.load();

            const promise =
                player.play();

            if (
                promise &&
                promise.catch
            ) {{
                promise.catch(
                    () => {{}}
                );
            }}
        }}


        player.addEventListener(
            "ended",
            () => {{

                index++;

                if (
                    index <
                    sequence.length
                ) {{
                    playCurrent();
                }}
                else {{
                    index = 0;
                    playCurrent();
                }}
            }}
        );


        buildProgress();

        playCurrent();

    </script>

    </body>

    </html>
    """

    components.html(
        page,
        height=height,
        scrolling=False,
    )

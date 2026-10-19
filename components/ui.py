from pathlib import Path
import streamlit as st


def theme() -> None:
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
    h1,h2,h3 {font-family:'Manrope',sans-serif; letter-spacing:-.035em;}
    .hero {padding:2.6rem 2.6rem 2.1rem;border:1px solid #8b5cf640;border-radius:24px;
      background:linear-gradient(125deg,#8b5cf617,#0eaaa010);margin-bottom:1.6rem;}
    .eyebrow {font-size:.74rem;letter-spacing:.2em;font-weight:700;color:#8b5cf6;text-transform:uppercase;}
    .hero h1 {font-size:3.25rem;line-height:1.08;margin:.8rem 0;max-width:900px;}
    .hero p {font-size:1.02rem;opacity:.78;max-width:760px;line-height:1.7;}
    .species-card {padding:1.3rem;border:1px solid #88888830;border-radius:18px;height:100%;}
    .species-card h3 {margin:.6rem 0;}
    .species-number {font-family:'Manrope',sans-serif;font-size:2rem;opacity:.25;}
    .workflow {border-left:3px solid #8b5cf6;padding:.6rem 1rem;margin:.7rem 0;background:#8b5cf609;}
    .footer {font-size:.77rem;opacity:.6;border-top:1px solid #88888830;padding-top:1rem;margin-top:3rem;}
    @media(max-width:640px) {.hero{padding:1.4rem}.hero h1{font-size:2.1rem}}
    </style>""", unsafe_allow_html=True)


def hero(eyebrow: str, title: str, body: str) -> None:
    st.markdown(f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{body}</p></div>', unsafe_allow_html=True)


def chart(fig, key: str, download: bool = False) -> None:
    st.plotly_chart(fig, use_container_width=True, key=key)
    if download:
        st.download_button("Download interactive figure", fig.to_html(include_plotlyjs=True), file_name=f"{key}.html", mime="text/html", key=f"dl_{key}")


def file_download(path: Path, label: str) -> None:
    if path.exists():
        st.download_button(label, path.read_bytes(), file_name=path.name, key=f"file_{path.name}")

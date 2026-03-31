# 2D is fine on runtime image; we still enable GPUs for future 3D
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# System libs for Qt/X11 + tools; include all xcb bits so the xcb platform plugin loads cleanly.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-pip wget ca-certificates \
    libglib2.0-0 libdbus-1-3 \
    libx11-6 libx11-xcb1 libxrender1 \
    libxcb1 libxcb-render0 libxcb-shm0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
    libxcb-randr0 libxcb-xfixes0 libxcb-xinerama0 libxcb-cursor0 libxcb-shape0 libxcb-xkb1 libxcb-render-util0 \
    libxrandr2 libxcursor1 libxcomposite1 libxdamage1 \
    libxkbcommon0 libxkbcommon-x11-0 \
    libgl1 libglu1-mesa \
    libfontconfig1 libsm6 libice6 libxau6 libxdmcp6 \
    fonts-dejavu-core xauth mesa-utils \
 && rm -rf /var/lib/apt/lists/*


# (Optional) VirtualGL for future true 3D/VTK; harmless for 2D.
# If SourceForge URL changes, bump version here.
# RUN wget -O /tmp/virtualgl.deb \
#       https://sourceforge.net/projects/virtualgl/files/3.1.2/virtualgl_3.1.2_amd64.deb/download \
#  && dpkg -i /tmp/virtualgl.deb || (apt-get update && apt-get -f install -y && dpkg -i /tmp/virtualgl.deb) \
#  && rm -f /tmp/virtualgl.deb

# Python deps
RUN pip3 install --no-cache-dir PySide6 pyqtgraph nibabel numpy

# App
WORKDIR /app
# You can either COPY here or bind-mount at runtime. COPY for convenience:
# COPY simple_nii_viewer.py /app/simple_nii_viewer.py

# Helpful default to avoid odd Qt/GL integration paths
ENV QT_XCB_GL_INTEGRATION=none

CMD ["/bin/bash"]

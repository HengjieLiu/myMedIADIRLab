## Build the image

cd /home/hengjie/DL_projects/code_sync/myMedIADIRLab/viewer_deform
docker build -t medview:latest .


## TurboVNC viewer
To run:
    /opt/TurboVNC/bin/vncserver -localhost -geometry 1920x1080

    Will output a number e.g.
        Desktop 'TurboVNC: shenggpu8:1 (hengjie)' started on display shenggpu8:1
        1 -> 5901
    On windows VNC server: localhost:5901

Confirm TurboVNC version:
    /opt/TurboVNC/bin/vncserver -version
List running sessions:
    /opt/TurboVNC/bin/vncserver -list
Kill a session:
    /opt/TurboVNC/bin/vncserver -kill :1




export DISPLAY=:1  # your VNC display (host TurboVNC)

docker run --gpus all --rm -it \
  -e DISPLAY=${DISPLAY} \
  -e QT_XCB_GL_INTEGRATION=none \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v //data/hengjie/:/data \
  -v /home/hengjie/DL_projects/code_sync/myMedIADIRLab/viewer_deform:/app \
  medview:latest

Inside container run:
  apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      libdbus-1-3 \
      libx11-xcb1 libxcb1 libxcb-render0 libxcb-shm0 \
      libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
      libxcb-randr0 libxcb-xfixes0 libxcb-xinerama0 \
      libgl1 libglu1-mesa \
      libxkbcommon0 libxkbcommon-x11-0 \
      fonts-dejavu-core


    # sanity: should print ":1"
    echo $DISPLAY

    python3 /app/simple_nii_viewer.py --vol /data/wavereg/datasets/LUMIR_L2R/LUMIR25_val/imagesVal/LUMIRMRI_3454_0000.nii.gz

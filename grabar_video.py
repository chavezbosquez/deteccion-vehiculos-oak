'''
Base: examples/VideoEncoder/rgb_encoding
Graba video a partir de la cámara principal
Aporte:
  1) Crear automáticamente el archivo MP4 a partir del archivo h265
  2) Asignar el nombre del archivo a la fecha y hora actual (para evitar que se borre el video anterior)
Nota: Requiere ffmpeg instalado
'''
import depthai as dai
import os
from datetime import datetime
import locale
import cv2

# Crear pipeline
pipeline = dai.Pipeline()

# Entradas y salidas
camRgb = pipeline.create(dai.node.ColorCamera)
videoEnc = pipeline.create(dai.node.VideoEncoder)
xout = pipeline.create(dai.node.XLinkOut)

xout.setStreamName('h265')

# Propiedades
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
#OCB: camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
videoEnc.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.H265_MAIN)

# Enlaces
camRgb.video.link(videoEnc.input)
videoEnc.bitstream.link(xout.input)

# OCB: Mostrar preview del video
xoutPreview = pipeline.create(dai.node.XLinkOut)    # Entradas y salidas
xoutPreview.setStreamName("preview")
camRgb.setPreviewSize(1280, 720)                     # Propiedades
camRgb.preview.link(xoutPreview.input)               # Enlace

print(camRgb.getResolution())

# Conectarse al dispositivo e iniciar pipeline
with dai.Device(pipeline) as device:
    # Cuantas cámaras
    print('Cámaras conectadas:', device.getConnectedCameras())
    # Velocidad USB
    print('Velocidad USB:', device.getUsbSpeed().name)

    # Cola de salida
    q = device.getOutputQueue(name="h265", maxSize=30, blocking=True)
    # OCB: Mostrar preview del video
    preview = device.getOutputQueue('preview')

    # El archivo .h265 es un 'raw stream' (no es reproducible)
    with open('video.h265', 'wb') as videoFile:
        print("Press Ctrl+C to stop encoding...")
        try:
            while True:
                h265Packet = q.get()  # Llamada de boqueo, espera hasta que un nuevo dato llega
                h265Packet.getData().tofile(videoFile)  # Agrega el paquete de dato al archivo abierto

                # OCB: Mostrar preview el video
                previewFrame = preview.get()
                cv2.imshow("Preview en vivo", previewFrame.getFrame())  # Muestra en frame 'preview'
                if cv2.waitKey(1) == ord('q'):  # Terminar la app presionando la tecla 'Q'
                    break

        except KeyboardInterrupt:
            pass    # Interrupción del teclado (Ctrl+C)

    # Crear video MP4
    locale.setlocale(locale.LC_TIME, "")
    ahora = datetime.now()
    nombre_archivo = ahora.strftime('%d-%B-%Y-%H_%M_%S') + '.mp4'
    #os.system("ffmpeg -framerate 30 -i video.h265 -c copy video.mp4")
    comando = 'ffmpeg -framerate 30 -i video.h265 -c copy ' + nombre_archivo
    os.system(comando)
    os.system('rm video.h265')
    print(comando)

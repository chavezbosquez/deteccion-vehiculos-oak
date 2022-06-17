# Módulos necesarios
from pathlib import Path
import blobconverter
import cv2
import depthai
import numpy as np

# El objeto Pipeline define las operaciones a realizar cuando se ejecute DepthAI
pipeline = depthai.Pipeline()

# Solo utilizar la camara central (a Color)
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)  # 300x300 será el tamaño de previsualización de cada frame, disponible como salida del nodo
#cam_rgb.setPreviewKeepAspectRatio(False)
cam_rgb.setInterleaved(False)

# Red neuronal para hacer las detecciones
detection_nn = pipeline.createMobileNetDetectionNetwork()
detection_nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
detection_nn.setConfidenceThreshold(0.01)
cam_rgb.preview.link(detection_nn.input)

xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

azul = (255, 0, 0)
verde = (36, 255, 12)
rojo = (0, 0, 255)
blanco = (255, 255, 255)
negro  = (0, 0, 0)
titulo = "Prueba de laboratorio"

def displayFrame(frame):
    for detection in detections:
        # Autos
        if detection.label == 7:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            #cv2.putText(frame, "Auto", (bbox[0]+10, bbox[1]+20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            #cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), azul, 2)                            # Ventana 300x300:  #0.6        #2
            cv2.putText(frame, 'Auto ' + f"{int(detection.confidence * 100)}%", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, verde, 1)
        # Bicicleta
        if detection.label == 2:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), rojo, 2)
            cv2.putText(frame, 'Bici ' + f"{int(detection.confidence * 100)}%", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, azul, 1)
        # Autobus
        if detection.label == 6:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), blanco, 2)
            cv2.putText(frame, 'Autobus ' + f"{int(detection.confidence * 100)}%", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, negro, 1)
        # Motocicleta
        if detection.label == 14:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), negro, 2)
            cv2.putText(frame, 'Motocicleta ' + f"{int(detection.confidence * 100)}%", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blanco, 1)
    # Show the frame
    cv2.imshow(titulo, frame)

cv2.namedWindow(titulo, cv2.WINDOW_NORMAL)
cv2.resizeWindow(titulo, 600,600)

with depthai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue("rgb")
    q_nn  = device.getOutputQueue("nn")
    frame = None
    detections = []

    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
    while True:
        in_rgb = q_rgb.tryGet()
        in_nn  = q_nn.tryGet()
        if in_rgb is not None:
            frame = in_rgb.getCvFrame()

        if in_nn is not None:
            detections = in_nn.detections

        if frame is not None:
            '''
            for detection in detections:
                # for each bounding box, we first normalize it to match the frame size
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                # and then draw a rectangle on the frame to show the actual result
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                ##if detection.
                cv2.putText(frame, 'Auto', (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), cv2.LINE_4)
            #OCB
            ####cv2.namedWindow("Prueba de laboratorio", cv2.WINDOW_FULLSCREEN)
            ####frame = cv2.resize(frame, (600, 600))
            # After all the drawing is finished, we show the frame on the screen
            cv2.imshow("Prueba de laboratorio", frame)
            '''
            displayFrame(frame)
        if cv2.waitKey(1) == ord('q'):
            break

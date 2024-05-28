import React, { useEffect, useRef } from "react";
import * as faceapi from "face-api.js";

const FaceDetection: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const loadModels = async () => {
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri("/models"),
        faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
        faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
        faceapi.nets.faceExpressionNet.loadFromUri("/models"),
      ]);
      startVideo();
    };

    const startVideo = () => {
      navigator.mediaDevices
        .getUserMedia({ video: {} })
        .then((stream) => {
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            videoRef.current.onloadedmetadata = () => {
              videoRef.current?.play();
              handleVideoPlay();
            };
          }
        })
        .catch((err) => console.error("Error accessing webcam:", err));
    };

    loadModels();
  }, []);

  const handleVideoPlay = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const displaySize = {
        width: video.videoWidth,
        height: video.videoHeight,
      };

      // Ensure canvas matches video dimensions
      canvas.width = displaySize.width;
      canvas.height = displaySize.height;
      faceapi.matchDimensions(canvas, displaySize);

      const updateDetections = async () => {
        const detections = await faceapi
          .detectAllFaces(
            video,
            new faceapi.TinyFaceDetectorOptions({
              inputSize: 512,
              scoreThreshold: 0.5,
            })
          )
          .withFaceLandmarks()
          .withFaceExpressions();

        const resizedDetections = faceapi.resizeResults(
          detections,
          displaySize
        );

        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          faceapi.draw.drawDetections(canvas, resizedDetections);
          faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
          faceapi.draw.drawFaceExpressions(canvas, resizedDetections);

          if (resizedDetections.length > 0) {
            const landmarks = resizedDetections[0].landmarks;
            drawLandmarkDistances(ctx, landmarks);
          }
        }
      };

      // Use requestAnimationFrame for smoother updates
      const tick = () => {
        updateDetections();
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    }
  };

  const getCenterPoint = (points: faceapi.Point[]) => {
    const sum = points.reduce(
      (acc, point) => ({
        x: acc.x + point.x,
        y: acc.y + point.y,
      }),
      { x: 0, y: 0 }
    );

    return {
      x: sum.x / points.length,
      y: sum.y / points.length,
    };
  };

  const drawLandmarkDistances = (
    ctx: CanvasRenderingContext2D,
    landmarks: faceapi.FaceLandmarks68
  ) => {
    ctx.fillStyle = "blue";
    ctx.font = "12px Arial";

    const keyPoints = [
      { name: "Left Eye", points: landmarks.getLeftEye() },
      { name: "Right Eye", points: landmarks.getRightEye() },
      { name: "Nose", points: landmarks.getNose() },
      { name: "Mouth", points: landmarks.getMouth() },
      { name: "Jawline", points: landmarks.getJawOutline() },
      { name: "Left Eyebrow", points: landmarks.getLeftEyeBrow() },
      { name: "Right Eyebrow", points: landmarks.getRightEyeBrow() },
    ];

    const centers = keyPoints.map((keyPoint) => ({
      name: keyPoint.name,
      center: getCenterPoint(keyPoint.points),
    }));

    // Draw distances between specific pairs of points
    for (let i = 0; i < centers.length; i++) {
      for (let j = i + 1; j < centers.length; j++) {
        const distance = Math.sqrt(
          Math.pow(centers[i].center.x - centers[j].center.x, 2) +
            Math.pow(centers[i].center.y - centers[j].center.y, 2)
        );
        const midPoint = {
          x: (centers[i].center.x + centers[j].center.x) / 2,
          y: (centers[i].center.y + centers[j].center.y) / 2,
        };
        ctx.fillText(`${distance.toFixed(1)}`, midPoint.x, midPoint.y);
      }
    }
  };

  return (
    <div style={{ position: "relative", width: "720px", height: "560px" }}>
      <video
        ref={videoRef}
        autoPlay
        muted
        style={{ width: "100%", height: "100%" }}
      />
      <canvas
        ref={canvasRef}
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
        }}
      />
    </div>
  );
};

export default FaceDetection;

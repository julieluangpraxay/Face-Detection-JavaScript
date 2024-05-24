// const video = document.getElementById('video')

// Promise.all([
//   faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
//   faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
//   faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
//   faceapi.nets.faceExpressionNet.loadFromUri('/models')
// ]).then(startVideo)

// function startVideo() {
//   navigator.getUserMedia(
//     { video: {} },
//     stream => video.srcObject = stream,
//     err => console.error(err)
//   )
// }

// video.addEventListener('play', () => {
//   const canvas = faceapi.createCanvasFromMedia(video)
//   document.body.append(canvas)
//   const displaySize = { width: video.width, height: video.height }
//   faceapi.matchDimensions(canvas, displaySize)
//   setInterval(async () => {
//     const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceExpressions()
//     const resizedDetections = faceapi.resizeResults(detections, displaySize)
//     canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
//     faceapi.draw.drawDetections(canvas, resizedDetections)
//     faceapi.draw.drawFaceLandmarks(canvas, resizedDetections)
//     faceapi.draw.drawFaceExpressions(canvas, resizedDetections)
//   }, 100)
// })

// code above for regular

const video = document.getElementById('video');

Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
  faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
  faceapi.nets.faceExpressionNet.loadFromUri('/models')
]).then(startVideo);

function startVideo() {
  navigator.getUserMedia(
    { video: {} },
    stream => video.srcObject = stream,
    err => console.error(err)
  );
}

video.addEventListener('play', () => {
  const canvas = faceapi.createCanvasFromMedia(video);
  document.body.append(canvas);
  const displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);
  setInterval(async () => {
    const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceExpressions();
    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    faceapi.draw.drawDetections(canvas, resizedDetections);
    faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
    faceapi.draw.drawFaceExpressions(canvas, resizedDetections);

    if (resizedDetections.length > 0) {
      const landmarks = resizedDetections[0].landmarks;
      drawLandmarkCoordinates(ctx, landmarks);
    }
  }, 100);
});

function drawLandmarkCoordinates(ctx, landmarks) {
  ctx.fillStyle = 'blue';
  ctx.font = '12px Arial';

  const points = landmarks.positions;
  points.forEach((point, index) => {
    ctx.fillText(`${index}`, point.x, point.y);
    ctx.fillText(`(${point.x.toFixed(1)}, ${point.y.toFixed(1)})`, point.x + 5, point.y - 5);
  });
}

function toggleVideo() {
  const video = document.getElementById('video');
  const button = document.getElementById('toggleButton');
  if (video.style.display === 'none') {
    video.style.display = 'block';
    button.textContent = 'Hide Video';
  } else {
    video.style.display = 'none';
    button.textContent = 'Show Video';
  }
}

document.getElementById('toggleButton').addEventListener('click', toggleVideo);

// Capturando a webcam;
const video = document.getElementById('video')
function startVideo() {
    navigator.getUserMedia(
      { video: {} },
      stream => video.srcObject = stream,
      err => console.error(err)
    )
  }

  //carregando as imagens nas pastas
  const loadLabels = () => {
    const labels = ['Alex','Luidy']
    return Promise.all(labels.map(async label => {
      const descriptions = [];
      for(let i=1; i<=2; i++) {
        const imgToRecognize = await faceapi.fetchImage(`../labels/${label}/${i}.png`)
        const detections = await faceapi
          .detectSingleFace(imgToRecognize)
          .withFaceLandmarks()
          .withFaceDescriptor()
        descriptions.push(detections.descriptor);
      }
      return new faceapi.LabeledFaceDescriptors(label, descriptions)
    }))
  }



  // carregando os modelos da api e iniciando o vídeo
  Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri('../models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('../models'),
    faceapi.nets.faceRecognitionNet.loadFromUri('../models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('../models')
  ]).then(startVideo)

  //Capturando a webcam e analisando com ferramentas da Api.
  video.addEventListener('play', async () => {
    const canvas = faceapi.createCanvasFromMedia(video)
    const canvasSize = {
      width:video.width,
      height:video.height
    }
    const labels = await loadLabels()
    faceapi.matchDimensions(canvas, canvasSize)
    document.body.appendChild(canvas)
    setInterval(async () => {
      const detections = await faceapi
        .detectAllFaces(
          video,
          new faceapi.TinyFaceDetectorOptions()
          )
         .withFaceLandmarks()
         .withFaceDescriptors()
         const resizedDetections = faceapi.resizeResults(detections, canvasSize)
         const faceMatcher = new faceapi.FaceMatcher(labels, 0.6)  
        const results = resizedDetections.map(detected => 
          faceMatcher.findBestMatch(detected.descriptor)
          )
         canvas.getContext('2d').clearRect(0,0, canvas.width, canvas.height)
         faceapi.draw.drawDetections(canvas,resizedDetections)
         faceapi.draw.drawFaceLandmarks(canvas,resizedDetections)
         results.forEach((result, index)=> {
          const box = resizedDetections[index].detection.box
          const {label, distance} = result
          new faceapi.draw.DrawTextField([
            `${label} (${parseInt(distance*100,10)})`
          ], box.bottomRight).draw(canvas)
    })
    
  },100)
  })
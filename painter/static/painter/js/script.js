// DOM Elements
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const video = document.getElementById('video');

// Event listener to start the webcam
startBtn.addEventListener('click', () => {
    // Enable the stop button and disable the start button
    startBtn.disabled = true;
    stopBtn.disabled = false;
    
    // Start the webcam feed (mocked with img/video source in this case)
    video.src = "/video_feed/";  // In a real app, this might trigger webcam streaming

    console.log('Webcam started');
});

// Event listener to stop the webcam
stopBtn.addEventListener('click', () => {
    // Enable the start button and disable the stop button
    startBtn.disabled = false;
    stopBtn.disabled = true;

    // Clear the video feed (simulating stop functionality)
    video.src = "";

    console.log('Webcam stopped');
});

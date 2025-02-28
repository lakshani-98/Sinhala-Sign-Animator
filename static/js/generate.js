document.getElementById('translateForm').addEventListener('submit', function(event) {
    event.preventDefault();
    var formData = new FormData(this);
    var loadingIndicator = document.getElementById('loadingIndicator');
    var loadingIndicator2 = document.getElementById('loadingIndicator2');
    loadingIndicator.style.display = 'block';
    loadingIndicator2.style.display = 'block';
    var outputVideoSection = document.getElementById('explore');
    outputVideoSection.scrollIntoView({ behavior: 'smooth' });

    fetch('/translate', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        var video_name = data.video_name;
        var videoSource = document.getElementById('videoSource');
        videoSource.src = "output/"+video_name;
        // Load and display the video tag
        var outputVideo = document.getElementById('outputVideo');
        outputVideo.load();
        outputVideo.style.display = 'block';
        loadingIndicator.style.display = 'none';
        loadingIndicator2.style.display = 'none';
        outputVideo.scrollIntoView({ behavior: 'smooth' });
    })
    .catch(error => {
        console.error('Error:', error);
        loadingIndicator.style.display = 'none';
        loadingIndicator2.style.display = 'none';
    });
});
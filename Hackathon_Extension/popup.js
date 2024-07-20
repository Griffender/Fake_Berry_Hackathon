document.getElementById('openApp').addEventListener('click', function() {
  chrome.tabs.create({ url: 'https://fakeberry-hackathon.streamlit.app/#ai-vs-human-text-classification' });
});

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "classifyText",
    title: "Classify Text",
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "classifyText") {
    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: classifySelectedText,
      args: [info.selectionText]
    });
  }
});

async function classifySelectedText(selectedText) {
  const url = "https://b784-13-53-64-97.ngrok-free.app/verify_and_check_bias";
  const threshold = 0.5;  // Default threshold value

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: selectedText, ai_score_threshold: threshold })
  });

  if (response.ok) {
    const result = await response.json();
    const classification = result.classification || "N/A";
    const probability_of_toxicity = result.probability_of_toxicity || 0.0;
    const prediction = result.prediction || "N/A";

    alert(`Classification: ${classification}\nPrediction: ${prediction}\nToxicity: ${Math.round(probability_of_toxicity * 100)}%`);
  } else {
    alert("Error: Unable to classify the text. Please try again later.");
  }
}

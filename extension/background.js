chrome.runtime.onMessage.addListener(
  function(request, sender, sendResponse) {
    console.log(sender.tab ?
      "from a content script:" + sender.tab.url :
      "from the extension");

    let text = request.sentence

    fetch('http://facebook.net:5000', {method: 'POST',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({sentence: text})})
      .then(res => {
        console.log(res.body);
        return res.json();
      })
      .then(json => {
        sendResponse(json);
      });
    return true;
  });

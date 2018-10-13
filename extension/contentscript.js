console.log('hello v2');


window.onkeypress = function () {
  const text = document.activeElement.textContent;
  if (text.length > 0) {
    _.debounce(function() {
      // Make ajax call
      // pretend like there's a response rn lol.
    }, 100)
  }
}

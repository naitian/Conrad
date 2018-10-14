console.log('hello v2');


const love_icon = '<i class="_3j7m _2p78 _9-- _hly"></i>';
const haha_icon = '<i class="_3j7o _2p78 _9-- _hly"></i>';
const sad_icon = '<i class="_3j7r _2p78 _9-- _hly"></i>';
const wow_icon = '<i class="_3j7n _2p78 _9-- _hly"></i>';
const angry_icon = '<i class="_3j7q _2p78 _9-- _hly"></i>';

const icon_map = {
  'love': love_icon,
  'haha': haha_icon,
  'wow': wow_icon,
  'sad': sad_icon,
  'angry': angry_icon
}

function generate_display(res) {
  let tpl = `
    <i class="_3j7m _2p78 _9-- _hly"></i> <span>${res.love} </span>
    <i class="_3j7o _2p78 _9-- _hly"></i> <span>${res.haha} </span>
    <i class="_3j7r _2p78 _9-- _hly"></i> <span>${res.wow} </span>
    <i class="_3j7n _2p78 _9-- _hly"></i> <span>${res.sad} </span>
    <i class="_3j7q _2p78 _9-- _hly"></i> <span>${res.angry} </span>
    `
  return tpl;
}

window.onkeypress = _.debounce(function() {
  const active = document.activeElement;
  const text = active.textContent;

  const display_thing = document.body.querySelector('.conrad-display-thing');
  let div;

  if (!display_thing) {
    div = document.createElement('div');
    div.classList.add("conrad-display-thing")
  } else {
    div = display_thing;
    div.innerHTML = '';
  }
  if (text.length > 0) {
    // Make ajax call
    // pretend like there's a response rn lol.
    console.log('hellooo');
    const res = {
      'love': 3,
      'haha': 5,
      'wow': 2,
      'sad': 6,
      'angry': 2
    };

    div.innerHTML = generate_display(res);
    x = active.getBoundingClientRect().left;
    y = active.getBoundingClientRect().bottom + 10;
    div.style.left = `${x}px`;
    div.style.top = `${y}px`;
    document.body.appendChild(div);
  }
}, 400)

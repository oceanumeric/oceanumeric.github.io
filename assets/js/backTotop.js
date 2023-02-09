// https://css-tricks.com/debouncing-throttling-explained-examples/
// https://stackoverflow.com/questions/31223341/detecting-scroll-direction
let scroll_position = 0;
let scroll_direction;
let scroll_up_count = 0;

window.addEventListener('scroll', function(e){
  if ((document.body.getBoundingClientRect()).top > scroll_position) {
    scroll_up_count ++;
  }
  scroll_position = (document.body.getBoundingClientRect()).top;

  setTimeout(() => {
    scroll_up_count = 0;
    let back_to_top = document.getElementById("back-top");
    back_to_top.setAttribute("hidden", "hidden");
  }, "6000")

  if (scroll_up_count > 10) {
    let back_to_top = document.getElementById("back-top");
    back_to_top.removeAttribute("hidden");
  }
});
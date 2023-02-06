# Filename: _plugins/compile_tags.rb
# Triggered once per build / serve session
Jekyll::Hooks.register :site, :after_init do
  system("python3 _plugins/blog_tag_generator.py")
  system("python3 _plugins/math_tag_generator.py")
end
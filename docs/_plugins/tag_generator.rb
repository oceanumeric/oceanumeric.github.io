# Filename: _plugins/compile_tags.rb
Jekyll::Hooks.register :posts, :post_write do
  system("python3 _plugins/blog_tag_generator.py")
end
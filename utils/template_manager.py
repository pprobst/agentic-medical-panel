"""Template management utilities."""

from pathlib import Path
from typing import Union
from jinja2 import Environment, FileSystemLoader


class TemplateManager:
    """Handles loading and rendering of Jinja2 templates."""

    def __init__(self, template_folder: Union[str, Path]):
        """
        Initializes the Jinja2 environment.

        Args:
            template_folder: The path to the directory containing template files.
        """
        self.jinja_env = Environment(
            loader=FileSystemLoader(template_folder), autoescape=True
        )

    def render(self, template_name: str, **context) -> str:
        """
        Renders a template with the given context.

        Args:
            template_name: The filename of the template to render.
            **context: Keyword arguments to pass to the template.

        Returns:
            The rendered template as a string.
        """
        template = self.jinja_env.get_template(template_name)
        return template.render(**context)

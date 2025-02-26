# Contributing to DeepSeek Document Chatbot

Thank you for your interest in contributing to the DeepSeek Document Chatbot! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. Any form of harassment or disrespectful behavior will not be tolerated.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with the following information:

- A clear, descriptive title
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Screenshots (if applicable)
- Environment information (OS, Python version, etc.)

### Suggesting Enhancements

We welcome suggestions for enhancing the chatbot. When suggesting enhancements, please include:

- A clear, descriptive title
- A detailed description of the proposed enhancement
- Any potential implementation details
- Why this enhancement would be useful

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Run tests to ensure your changes don't break existing functionality
5. Commit your changes (`git commit -m 'Add some feature'`)
6. Push to the branch (`git push origin feature/your-feature-name`)
7. Open a pull request

## Development Setup

1. Clone the repository
   ```bash
   git clone https://github.com/Egberun/deepseek-document-chatbot.git
   cd deepseek-document-chatbot
   ```

2. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Add your development documents to the `documents` directory

5. Run the application
   ```bash
   python main.py --cli
   ```

## Project Structure

- `document_processor.py`: Document loading and processing
- `llm_model.py`: DeepSeek LLM integration
- `retrieval_chain.py`: Conversational retrieval chain
- `main.py`: Main application
- `advanced_web_ui.py`: Web interface
- `config.py`: Configuration management
- `prompt_templates.py`: Prompt templates
- `monitoring.py`: Usage monitoring

## Coding Standards

- Follow PEP 8 style guidelines
- Write docstrings for all functions, classes, and modules
- Include type hints where appropriate
- Write unit tests for new functionality

## Testing

Before submitting a pull request, please ensure that your code passes all tests:

```bash
pytest
```

## Documentation

When adding new features, please update the relevant documentation. This includes:

- Code docstrings
- README.md (if applicable)
- Any other relevant documentation files

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (as specified in the LICENSE file).

## Questions?

If you have any questions about contributing, please feel free to open an issue.
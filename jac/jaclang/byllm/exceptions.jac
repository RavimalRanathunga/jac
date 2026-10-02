import from jaclang.byllm._optdeps.litellm { litellm }

obj ByLLMError(Exception) {
    has message: str = "";

    def postinit -> None;
}

impl ByLLMError.postinit -> None {
    Exception.__init__(self, self.message);
}

obj AuthenticationError(ByLLMError) {}

obj RateLimitError(ByLLMError) {}

obj ModelNotFoundError(ByLLMError) {}

obj LLMTimeout(ByLLMError, litellm.Timeout) {
    has model: str = "", llm_provider: str = "";

    def postinit -> None;
}

impl LLMTimeout.postinit -> None {
    Exception.__init__(self, self.message);
}

obj OutputConversionError(ByLLMError) {}

obj FinishToolError(ByLLMError) {}

obj ConfigurationError(ByLLMError) {}

obj McpError(ByLLMError) {}

obj CompactionNotEffectiveError(ByLLMError) {}

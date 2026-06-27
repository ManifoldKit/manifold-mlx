import XCTest
import ManifoldInference

/// CI gate: the MLX-relevant model family templates never silently drop a
/// declared tool dialect when rendered — the #1909 failure class.
///
/// `RenderConsistencyChecker.check(chatTemplateRaw:)` renders the Jinja template
/// with dummy tools and checks that the declared delimiter survives the render.
/// Catching a drop here costs nothing: no Apple Silicon, no GGUF, no inference.
///
/// Template constants mirror ManifoldKit's committed `RenderConsistencyCheckerTests`
/// corpus so the two corpora stay comparable — any template that passes there
/// must also pass here. See MK #2055 for the original gate introduction.
final class MLXRenderConsistencyGateTests: XCTestCase {

    // MARK: - Family template corpus (mirrors ManifoldKit's committed corpus)

    /// Qwen / Hermes `<tool_call>…</tool_call>` JSON dialect behind `{% if tools %}`.
    private static let qwenStyleTemplate = """
    {%- if tools %}
    <tools>
    {%- for tool in tools %}{{ tool.name }}
    {%- endfor %}
    </tools>
    {%- endif %}
    {%- for message in messages %}
    <|{{ message.role }}|>{{ message.content }}
    {%- if message.tool_calls %}
        {%- for tc in message.tool_calls %}<tool_call>
    {"name": "{{ tc.function.name }}"}
    </tool_call>
        {%- endfor %}
    {%- endif %}
    {%- endfor %}
    """

    /// Mistral-v0.3 `[TOOL_CALLS]` dialect behind `{% if tools is not none %}`.
    private static let mistralStyleTemplate = """
    {%- if tools is not none %}
    [AVAILABLE_TOOLS]
    {%- for tool in tools %}{{ tool.name }}
    {%- endfor %}
    [/AVAILABLE_TOOLS]
    {%- endif %}
    {%- for message in messages %}
    {{ message.role }}: {{ message.content }}
    {%- if message.tool_calls %}
        {%- for tc in message.tool_calls %}[TOOL_CALLS] {{ tc.function.name }}
        {%- endfor %}
    {%- endif %}
    {%- endfor %}
    """

    /// Hermes-style `<tool_call>` dialect behind a bare `{% for tool in tools %}`.
    private static let hermesStyleTemplate = """
    {%- for tool in tools %}{{ tool.name }}
    {%- endfor %}
    {%- for message in messages %}
    {{ message.role }}: {{ message.content }}
    {%- if message.tool_calls %}
        {%- for tc in message.tool_calls %}<tool_call>
    {"name": "{{ tc.function.name }}"}
    </tool_call>
        {%- endfor %}
    {%- endif %}
    {%- endfor %}
    """

    /// Gemma-style `<|tool_call>` key/value dialect behind `{% if tools %}`.
    private static let gemmaStyleTemplate = """
    {%- if tools %}
    Tools:
    {%- for tool in tools %}{{ tool.name }}
    {%- endfor %}
    {%- endif %}
    {%- for message in messages %}
    {{ message.role }}: {{ message.content }}
    {%- if message.tool_calls %}
        {%- for tc in message.tool_calls %}<|tool_call>
    name: {{ tc.function.name }}
    <|end_of_turn>
        {%- endfor %}
    {%- endif %}
    {%- endfor %}
    """

    /// Toolless (Phi-4 style): no `{% if tools %}` block at all.
    private static let toollessTemplate = """
    {%- for message in messages %}
    {{ message.role }}: {{ message.content }}
    {%- endfor %}
    """

    // MARK: - Gate test (the CI invariant)

    /// The invariant: no MLX-relevant tool-bearing family template returns
    /// `.inconsistent`. An `.inconsistent` result means the template *declared*
    /// a tool dialect but the renderer dropped it — the silent failure that
    /// causes zero tool calls without any error signal.
    func test_gate_knownGoodFamilyTemplates_neverRenderInconsistent() {
        let families: [(name: String, template: String)] = [
            ("qwen",    Self.qwenStyleTemplate),
            ("mistral", Self.mistralStyleTemplate),
            ("hermes",  Self.hermesStyleTemplate),
            ("gemma",   Self.gemmaStyleTemplate),
        ]
        for (name, template) in families {
            let result = RenderConsistencyChecker.check(chatTemplateRaw: template)
            XCTAssertNotEqual(
                result.status, .inconsistent,
                "\(name) family template returned .inconsistent — " +
                "the declared tool delimiter is being silently dropped by the renderer (#1909). " +
                "Run the full render golden suite to identify which dialect regressed."
            )
        }
    }

    // MARK: - Per-family status assertions

    func test_qwenTemplate_isConsistent() {
        XCTAssertEqual(
            RenderConsistencyChecker.check(chatTemplateRaw: Self.qwenStyleTemplate).status,
            .consistent
        )
    }

    func test_mistralTemplate_isConsistent() {
        XCTAssertEqual(
            RenderConsistencyChecker.check(chatTemplateRaw: Self.mistralStyleTemplate).status,
            .consistent
        )
    }

    func test_hermesTemplate_isConsistent() {
        XCTAssertEqual(
            RenderConsistencyChecker.check(chatTemplateRaw: Self.hermesStyleTemplate).status,
            .consistent
        )
    }

    func test_gemmaTemplate_isConsistent() {
        XCTAssertEqual(
            RenderConsistencyChecker.check(chatTemplateRaw: Self.gemmaStyleTemplate).status,
            .consistent
        )
    }

    func test_toollessTemplate_isNotApplicable() {
        XCTAssertEqual(
            RenderConsistencyChecker.check(chatTemplateRaw: Self.toollessTemplate).status,
            .notApplicable
        )
    }
}

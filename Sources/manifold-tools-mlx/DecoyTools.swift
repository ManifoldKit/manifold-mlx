// DecoyTools — distractor tools for the tool-selection stress harness.
//
// The scenario harness normally advertises ONLY each scenario's
// `requiredTools`, so a passing run can't distinguish "the model picked the
// right tool" from "the model had no other tool to pick." To measure
// correct-tool *selection* we pad the advertised toolset with N decoys
// (`--extra-tools N`) and check that the model still invokes the required
// tool and never reaches for a decoy.
//
// The pool is deliberately drawn from assistant domains ORTHOGONAL to the six
// reference tools (time / calc / file / dir / repo-search / http-fixture):
// none of these is ever a legitimate alternative for a built-in scenario, so a
// decoy invocation is unambiguously a wrong-tool selection rather than a
// defensible substitution. Names and schemas are realistic so they exert real
// distractor pressure — a model that pattern-matches on "looks like a tool I
// know" should still be pulled toward them.
//
// Selection is the first N of a fixed, ordered pool (no RNG — runs must be
// reproducible and diffable, matching the harness's deterministic-replay
// contract). The pool holds 24 tools, covering N up to 20 with headroom.
import Foundation
import ManifoldInference
import ManifoldTools

enum DecoyTools {

    /// Fixed result shape for every decoy. The content is inert — decoys exist
    /// to be *advertised*, and if a model wrongly calls one we want a benign,
    /// recognisable payload in the transcript rather than an error.
    struct Result: Encodable, Sendable {
        let status: String
    }

    /// One entry in the decoy pool: the advertised contract plus a stub result.
    private struct Spec {
        let name: String
        let description: String
        /// Property name → human description. All properties are advertised as
        /// required strings; the executor ignores them (EmptyArgs is permissive),
        /// so the schema can look real without constraining dispatch.
        let properties: [(String, String)]
    }

    /// The ordered distractor pool. Order is the selection order for `--extra-tools N`.
    private static let pool: [Spec] = [
        Spec(name: "get_weather", description: "Returns the current weather for a city. Call when the user asks about weather, temperature, or forecast.",
             properties: [("city", "City name, e.g. 'San Francisco'."), ("units", "'metric' or 'imperial'.")]),
        Spec(name: "convert_currency", description: "Converts an amount between two currencies at the current exchange rate.",
             properties: [("amount", "Amount to convert."), ("from", "Source ISO currency code."), ("to", "Target ISO currency code.")]),
        Spec(name: "send_email", description: "Sends an email on the user's behalf. Call only when the user explicitly asks to send mail.",
             properties: [("to", "Recipient address."), ("subject", "Subject line."), ("body", "Message body.")]),
        Spec(name: "translate_text", description: "Translates text into a target language.",
             properties: [("text", "Text to translate."), ("target_lang", "Target language code, e.g. 'es'.")]),
        Spec(name: "set_timer", description: "Starts a countdown timer for a number of seconds.",
             properties: [("seconds", "Duration in seconds."), ("label", "Optional label for the timer.")]),
        Spec(name: "create_calendar_event", description: "Creates an event on the user's calendar.",
             properties: [("title", "Event title."), ("start", "ISO-8601 start time."), ("end", "ISO-8601 end time.")]),
        Spec(name: "get_stock_price", description: "Returns the latest trading price for a stock ticker.",
             properties: [("symbol", "Ticker symbol, e.g. 'AAPL'.")]),
        Spec(name: "play_music", description: "Plays a track from the user's music library.",
             properties: [("track", "Track name."), ("artist", "Artist name.")]),
        Spec(name: "send_sms", description: "Sends an SMS text message.",
             properties: [("phone", "Recipient phone number."), ("message", "Message text.")]),
        Spec(name: "book_flight", description: "Books a flight between two airports on a given date.",
             properties: [("origin", "Origin airport code."), ("destination", "Destination airport code."), ("date", "Departure date.")]),
        Spec(name: "get_directions", description: "Returns turn-by-turn directions between two locations.",
             properties: [("origin", "Start location."), ("destination", "End location.")]),
        Spec(name: "create_reminder", description: "Creates a reminder due at a given time.",
             properties: [("text", "Reminder text."), ("due", "ISO-8601 due time.")]),
        Spec(name: "post_social_update", description: "Posts a status update to the user's social account.",
             properties: [("message", "Status text.")]),
        Spec(name: "get_news_headlines", description: "Returns recent news headlines for a topic.",
             properties: [("topic", "Topic or keyword.")]),
        Spec(name: "control_smart_light", description: "Turns a smart light on or off in a given room.",
             properties: [("room", "Room name."), ("state", "'on' or 'off'.")]),
        Spec(name: "order_food", description: "Places a food delivery order from a restaurant.",
             properties: [("restaurant", "Restaurant name."), ("items", "Comma-separated items.")]),
        Spec(name: "start_workout", description: "Starts tracking a workout session.",
             properties: [("type", "Workout type, e.g. 'run'."), ("duration", "Planned duration in minutes.")]),
        Spec(name: "scan_qr_code", description: "Decodes a QR code from an image.",
             properties: [("image", "Path or identifier of the image.")]),
        Spec(name: "lookup_dictionary", description: "Returns the dictionary definition of a word.",
             properties: [("word", "Word to define.")]),
        Spec(name: "get_traffic", description: "Returns current traffic conditions along a route.",
             properties: [("route", "Route or road name.")]),
        Spec(name: "set_thermostat", description: "Sets the home thermostat to a target temperature.",
             properties: [("temperature", "Target temperature in degrees.")]),
        Spec(name: "find_parking", description: "Finds available parking near a location.",
             properties: [("location", "Location to search near.")]),
        Spec(name: "get_sunrise_sunset", description: "Returns sunrise and sunset times for a location and date.",
             properties: [("location", "Location name."), ("date", "Date in YYYY-MM-DD.")]),
        Spec(name: "track_package", description: "Returns the delivery status for a shipment tracking number.",
             properties: [("tracking_number", "Carrier tracking number.")]),
    ]

    /// Largest `--extra-tools N` the pool can satisfy.
    static var maxCount: Int { pool.count }

    /// Names of the first `count` decoys, in selection order. Used to augment a
    /// scenario's `requiredTools` so `ScenarioRunner`'s advertise-filter lets
    /// them through, and to detect wrong-tool selections after a run.
    static func names(count: Int) -> [String] {
        pool.prefix(max(0, count)).map(\.name)
    }

    /// Executors for the first `count` decoys, ready to register on a
    /// `ToolRegistry`. Each ignores its arguments and returns an inert result.
    static func executors(count: Int) -> [any ToolExecutor] {
        pool.prefix(max(0, count)).map { spec in
            let definition = ToolDefinition(
                name: spec.name,
                description: spec.description,
                parameters: schema(for: spec.properties)
            )
            return TypedToolExecutor<EmptyArgs, Result>(definition: definition) { _ in
                Result(status: "ok")
            }
        }
    }

    /// Builds a JSON-Schema object advertising every property as a required string.
    private static func schema(for properties: [(String, String)]) -> JSONSchemaValue {
        var props: [String: JSONSchemaValue] = [:]
        var required: [JSONSchemaValue] = []
        for (name, desc) in properties {
            props[name] = .object([
                "type": .string("string"),
                "description": .string(desc)
            ])
            required.append(.string(name))
        }
        return .object([
            "type": .string("object"),
            "properties": .object(props),
            "required": .array(required)
        ])
    }
}

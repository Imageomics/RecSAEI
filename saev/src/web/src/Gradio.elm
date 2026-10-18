module Gradio exposing
    ( Base64Image
    , Config
    , Error(..)
    , HttpUrlImage
    , base64Image
    , base64ImageDecoder
    , base64ImageEmpty
    , base64ImageToString
    , decodeOne
    , encodeImg
    , get
    , httpUrlImageDecoder
    )

import Http
import Json.Decode as D
import Json.Encode as E
import Parser exposing ((|.), (|=))
import Task
import Url.Builder


type alias Config =
    { host : String }


type Error
    = NetworkError String
    | ParsingError String
    | JsonError String
    | ApiError String
    | UserError String


type Base64Image
    = Base64Image String



-- Constructor function with validation if needed


base64Image : String -> Maybe Base64Image
base64Image str =
    if String.startsWith "data:image/" str then
        Just (Base64Image str)

    else
        Nothing


base64ImageToString : Base64Image -> String
base64ImageToString (Base64Image str) =
    str



-- Transparent 448x448 webp image. Used if you don't want to show any image.


base64ImageEmpty : Base64Image
base64ImageEmpty =
    Base64Image "data:image/webp;base64,UklGRtABAABXRUJQVlA4WAoAAAAQAAAAvwEAvwEAQUxQSBsAAAABBxAREVDQtg1T/vC744j+Z/jPf/7zn//8LwEAVlA4II4BAABQLQCdASrAAcABPpFIoU0lpCMiIAgAsBIJaW7hd2EbQAnsA99snIe+2TkPfbJyHvtk5D32ych77ZOQ99snIe+2TkPfbJyHvtk5D32ych77ZOQ99snIe+2TkPfbJyHvtk5D32ych77ZOQ99snIe+2TkPfbJyHvtk5D32ych77ZOQ99snIe+2TkPfbJyHvtk5D32ych77ZOQ99snIe+2TkPfbJyHvtk5D32ych77ZOQ99snIe+2TkPfbJyHvtk5D32ych77ZOQ99snIe+2TkPfbJyHvtk5D32ych77ZOQ99snIe+2TkPfbJyHvtk5D32ych77ZOQ99snIe+2TkPfbJyHvtk5D32ych77ZOQ99snIe+2TkPfbJyHvtk5D32ych77ZOQ99snIe+2TkPfbJyHvtk5D32ych77ZOQ99snIe+2TkPfbJyHvtk5D32ych77ZOQ99snIe+2TkPfbJyHvtk5D32ych77ZOQ99snIe+2TkPfasAD+/94IrMAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="


type HttpUrlImage
    = HttpUrlImage String


get : Config -> String -> List E.Value -> D.Decoder a -> (Result Error a -> msg) -> Cmd msg
get cfg path args decoder msg =
    start cfg path args
        |> Task.andThen (finish cfg path decoder)
        |> Task.attempt msg


start : Config -> String -> List E.Value -> Task.Task Error String
start cfg path args =
    Http.task
        { method = "POST"
        , headers = []
        , url =
            Url.Builder.crossOrigin cfg.host
                [ "gradio_api", "call", path ]
                []
        , body = Http.jsonBody (encodeArgs args)
        , resolver =
            Http.stringResolver
                (httpResolver
                    >> Result.andThen (jsonResolver eventIdDecoder)
                )
        , timeout = Nothing
        }


finish : Config -> String -> D.Decoder a -> String -> Task.Task Error a
finish cfg path decoder eventId =
    Http.task
        { method = "GET"
        , headers = []
        , url =
            Url.Builder.crossOrigin cfg.host
                [ "gradio_api", "call", path, eventId ]
                []
        , body = Http.emptyBody
        , resolver =
            Http.stringResolver
                (httpResolver
                    >> Result.andThen parsingResolver
                    >> Result.andThen (jsonResolver decoder)
                )
        , timeout = Nothing
        }


eventIdDecoder : D.Decoder String
eventIdDecoder =
    D.field "event_id" D.string


httpResolver : Http.Response String -> Result Error String
httpResolver response =
    case response of
        Http.GoodStatus_ _ body ->
            Ok body

        Http.BadUrl_ url ->
            Err (ApiError <| "Bad URL: " ++ url)

        Http.Timeout_ ->
            Err (NetworkError "Timed out")

        Http.NetworkError_ ->
            Err (NetworkError "Unknown network error")

        Http.BadStatus_ _ body ->
            Err (ApiError body)


parsingResolver : String -> Result Error String
parsingResolver raw =
    Parser.run eventParser raw
        |> Result.mapError (deadEndsToString >> ParsingError)


jsonResolver : D.Decoder a -> String -> Result Error a
jsonResolver decoder body =
    D.decodeString decoder body
        |> Result.mapError (D.errorToString >> JsonError)


encodeArgs : List E.Value -> E.Value
encodeArgs args =
    E.object [ ( "data", E.list identity args ) ]


encodeImg : Base64Image -> E.Value
encodeImg (Base64Image image) =
    E.object [ ( "url", E.string image ) ]


httpUrlImageDecoder : D.Decoder HttpUrlImage
httpUrlImageDecoder =
    D.field "url" D.string
        |> D.map (String.replace "gradio_api/gradio_api" "gradio_api")
        |> D.map (String.replace "gra/gradio_api" "gradio_api")
        |> D.map (String.replace "gradio_ap/gradio_api" "gradio_api")
        |> D.map (String.replace "gradio_api/ca/gradio_api" "gradio_api")
        |> D.map (String.replace "gradio_api/call/gradio_api" "gradio_api")
        |> D.map HttpUrlImage


base64ImageDecoder : D.Decoder Base64Image
base64ImageDecoder =
    D.string
        |> D.andThen
            (\str ->
                case base64Image str of
                    Just img ->
                        D.succeed img

                    Nothing ->
                        D.fail "Invalid base64 image format"
            )


decodeOne : D.Decoder a -> D.Decoder a
decodeOne decoder =
    D.list decoder
        |> D.map List.head
        |> D.andThen decodeOneHelper


decodeOneHelper : Maybe a -> D.Decoder a
decodeOneHelper maybe =
    case maybe of
        Just something ->
            D.succeed something

        Nothing ->
            D.fail "No result"



-- PARSER


eventParser : Parser.Parser String
eventParser =
    Parser.loop () eventParserHelper


eventParserHelper : () -> Parser.Parser (Parser.Step () String)
eventParserHelper _ =
    Parser.succeed identity
        |. Parser.keyword "event"
        |. Parser.symbol ":"
        |. Parser.spaces
        |= Parser.oneOf
            [ Parser.succeed (Parser.Loop ())
                |. Parser.keyword "heartbeat"
                |. Parser.spaces
                |. Parser.keyword "data"
                |. Parser.symbol ":"
                |. Parser.spaces
                |. Parser.keyword "null"
                |. Parser.spaces
            , Parser.succeed (\rest -> Parser.Done rest)
                |. Parser.keyword "complete"
                |. Parser.spaces
                |. Parser.keyword "data"
                |. Parser.symbol ":"
                |. Parser.spaces
                |= restParser
                |. Parser.spaces
                |. Parser.end
            ]


restParser : Parser.Parser String
restParser =
    Parser.getChompedString <|
        Parser.succeed ()
            |. Parser.chompUntilEndOr "\n"


deadEndsToString : List Parser.DeadEnd -> String
deadEndsToString deadEnds =
    String.concat (List.intersperse "; " (List.map deadEndToString deadEnds))


deadEndToString : Parser.DeadEnd -> String
deadEndToString deadend =
    problemToString deadend.problem ++ " at row " ++ String.fromInt deadend.row ++ ", col " ++ String.fromInt deadend.col


problemToString : Parser.Problem -> String
problemToString p =
    case p of
        Parser.Expecting s ->
            "expecting '" ++ s ++ "'"

        Parser.ExpectingInt ->
            "expecting int"

        Parser.ExpectingHex ->
            "expecting hex"

        Parser.ExpectingOctal ->
            "expecting octal"

        Parser.ExpectingBinary ->
            "expecting binary"

        Parser.ExpectingFloat ->
            "expecting float"

        Parser.ExpectingNumber ->
            "expecting number"

        Parser.ExpectingVariable ->
            "expecting variable"

        Parser.ExpectingSymbol s ->
            "expecting symbol '" ++ s ++ "'"

        Parser.ExpectingKeyword s ->
            "expecting keyword '" ++ s ++ "'"

        Parser.ExpectingEnd ->
            "expecting end"

        Parser.UnexpectedChar ->
            "unexpected char"

        Parser.Problem s ->
            "problem " ++ s

        Parser.BadRepeat ->
            "bad repeat"

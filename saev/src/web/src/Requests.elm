module Requests exposing
    ( Id
    , Requested(..)
    , Result
    , init
    , isStale
    , map
    , next
    )

-- MODEL


type Id
    = Id Int


init : Id
init =
    Id 0


type alias Result a =
    { id : Id
    , data : a
    }



-- UPDATE


next : Id -> Id
next (Id id) =
    Id (id + 1)


isStale : Id -> Id -> Bool
isStale (Id id) (Id last) =
    id < last



-- Requested


type Requested a e
    = Initial
    | Loading
    | Loaded a
    | Failed e


map : (a -> b) -> Requested a e -> Requested b e
map fn requested =
    case requested of
        Initial ->
            Initial

        Loading ->
            Loading

        Failed err ->
            Failed err

        Loaded a ->
            Loaded (fn a)

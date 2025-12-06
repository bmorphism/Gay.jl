# Self-hosted S-expression parser and evaluator for Gay.jl
#
# A minimal Lisp subset that handles the patterns we need without external dependencies.
# Designed to be robust where LispSyntax.jl has edge cases (e.g., floating point literals).
#
# Supported forms:
#   (+ 1 2)              → arithmetic
#   (defn name (args) body) → function definition
#   (let [x 1 y 2] body) → let bindings
#   (if cond then else)  → conditionals
#   (fn (args) body)     → anonymous functions
#   {:key val}           → dictionaries
#   [a b c]              → vectors
#   'symbol              → quoted symbols
#
# Design: Parse to Julia AST, then eval. No intermediate representation.

module SExp

# Julia character predicates
isalpha(c::Char) = isletter(c)
isalnum(c::Char) = isletter(c) || isdigit(c)

export @sx, sexp_parse, sexp_eval, sexp_read

# ═══════════════════════════════════════════════════════════════════════════════
# Tokenizer
# ═══════════════════════════════════════════════════════════════════════════════

struct Token
    type::Symbol  # :lparen, :rparen, :lbracket, :rbracket, :lbrace, :rbrace, :symbol, :number, :string, :keyword
    value::Any
    pos::Int
end

function tokenize(s::AbstractString)
    tokens = Token[]
    i = 1
    n = length(s)
    
    while i <= n
        c = s[i]
        
        # Whitespace
        if isspace(c)
            i += 1
            continue
        end
        
        # Comments (;)
        if c == ';'
            while i <= n && s[i] != '\n'
                i += 1
            end
            continue
        end
        
        # Delimiters
        if c == '('
            push!(tokens, Token(:lparen, '(', i))
            i += 1
        elseif c == ')'
            push!(tokens, Token(:rparen, ')', i))
            i += 1
        elseif c == '['
            push!(tokens, Token(:lbracket, '[', i))
            i += 1
        elseif c == ']'
            push!(tokens, Token(:rbracket, ']', i))
            i += 1
        elseif c == '{'
            push!(tokens, Token(:lbrace, '{', i))
            i += 1
        elseif c == '}'
            push!(tokens, Token(:rbrace, '}', i))
            i += 1
        elseif c == '\''
            push!(tokens, Token(:quote, '\'', i))
            i += 1
            
        # Strings
        elseif c == '"'
            start = i
            i += 1
            buf = IOBuffer()
            while i <= n && s[i] != '"'
                if s[i] == '\\' && i + 1 <= n
                    i += 1
                    esc_char = s[i]
                    if esc_char == 'n'
                        write(buf, '\n')
                    elseif esc_char == 't'
                        write(buf, '\t')
                    elseif esc_char == 'e'
                        write(buf, '\e')  # ANSI escape
                    elseif esc_char == '"'
                        write(buf, '"')
                    elseif esc_char == '\\'
                        write(buf, '\\')
                    else
                        write(buf, '\\', esc_char)
                    end
                else
                    write(buf, s[i])
                end
                i += 1
            end
            if i > n
                error("Unterminated string starting at position $start")
            end
            i += 1  # Skip closing "
            push!(tokens, Token(:string, String(take!(buf)), start))
            
        # Keywords (:keyword)
        elseif c == ':'
            start = i
            i += 1
            while i <= n && (isalnum(s[i]) || s[i] in "_-!?")
                i += 1
            end
            name = s[start+1:i-1]
            # Convert kebab-case to snake_case for Julia
            name = replace(name, "-" => "_")
            push!(tokens, Token(:keyword, Symbol(name), start))
            
        # Numbers (including floats)
        elseif isdigit(c) || (c == '-' && i + 1 <= n && isdigit(s[i+1]))
            start = i
            if c == '-'
                i += 1
            end
            while i <= n && isdigit(s[i])
                i += 1
            end
            # Check for decimal point
            if i <= n && s[i] == '.' && i + 1 <= n && isdigit(s[i+1])
                i += 1
                while i <= n && isdigit(s[i])
                    i += 1
                end
                # Check for exponent
                if i <= n && s[i] in "eE"
                    i += 1
                    if i <= n && s[i] in "+-"
                        i += 1
                    end
                    while i <= n && isdigit(s[i])
                        i += 1
                    end
                end
                push!(tokens, Token(:number, parse(Float64, s[start:i-1]), start))
            else
                push!(tokens, Token(:number, parse(Int, s[start:i-1]), start))
            end
            
        # Symbols
        elseif isalpha(c) || c in "_!?*+-/<>=&"
            start = i
            while i <= n && (isalnum(s[i]) || s[i] in "_-!?*+/<>=&.")
                i += 1
            end
            sym_str = s[start:i-1]
            # Handle special symbols
            if sym_str == "true"
                push!(tokens, Token(:bool, true, start))
            elseif sym_str == "false"
                push!(tokens, Token(:bool, false, start))
            elseif sym_str == "nil"
                push!(tokens, Token(:nil, nothing, start))
            else
                push!(tokens, Token(:symbol, Symbol(sym_str), start))
            end
        else
            error("Unexpected character '$c' at position $i")
        end
    end
    
    return tokens
end

# ═══════════════════════════════════════════════════════════════════════════════
# Parser: Tokens → S-expression (nested arrays/symbols/literals)
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct Parser
    tokens::Vector{Token}
    pos::Int
end

Parser(tokens) = Parser(tokens, 1)

function peek(p::Parser)
    p.pos <= length(p.tokens) ? p.tokens[p.pos] : nothing
end

function advance!(p::Parser)
    t = p.tokens[p.pos]
    p.pos += 1
    t
end

function parse_sexp(p::Parser)
    t = peek(p)
    t === nothing && return nothing
    
    if t.type == :lparen
        advance!(p)  # consume (
        items = Any[]
        while (t2 = peek(p)) !== nothing && t2.type != :rparen
            push!(items, parse_sexp(p))
        end
        if peek(p) === nothing
            error("Unclosed parenthesis")
        end
        advance!(p)  # consume )
        return items
        
    elseif t.type == :lbracket
        advance!(p)  # consume [
        items = Any[]
        while (t2 = peek(p)) !== nothing && t2.type != :rbracket
            push!(items, parse_sexp(p))
        end
        if peek(p) === nothing
            error("Unclosed bracket")
        end
        advance!(p)  # consume ]
        return (:vec, items)  # Mark as vector
        
    elseif t.type == :lbrace
        advance!(p)  # consume {
        items = Any[]
        while (t2 = peek(p)) !== nothing && t2.type != :rbrace
            push!(items, parse_sexp(p))
        end
        if peek(p) === nothing
            error("Unclosed brace")
        end
        advance!(p)  # consume }
        return (:dict, items)  # Mark as dict
        
    elseif t.type == :quote
        advance!(p)  # consume '
        return [:quote, parse_sexp(p)]
        
    elseif t.type in (:symbol, :number, :string, :keyword, :bool, :nil)
        advance!(p)
        return t.value
        
    else
        error("Unexpected token: $(t.type)")
    end
end

function sexp_read(s::AbstractString)
    tokens = tokenize(s)
    p = Parser(tokens)
    parse_sexp(p)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Code Generator: S-expression → Julia Expr
# ═══════════════════════════════════════════════════════════════════════════════

"""
Convert kebab-case symbol to Julia-compatible snake_case.
"""
function julia_sym(s::Symbol)
    str = string(s)
    # Keep ! and ? at end
    str = replace(str, "-" => "_")
    Symbol(str)
end

"""
Generate Julia AST from S-expression.
"""
function codegen(sexp)
    # Literals
    if sexp isa Number || sexp isa String || sexp isa Bool
        return sexp
    end
    
    if sexp === nothing
        return :nothing
    end
    
    # Symbols (convert kebab-case)
    if sexp isa Symbol
        return julia_sym(sexp)
    end
    
    # Vector literal
    if sexp isa Tuple && length(sexp) == 2 && sexp[1] == :vec
        items = sexp[2]
        return Expr(:vect, map(codegen, items)...)
    end
    
    # Dict literal
    if sexp isa Tuple && length(sexp) == 2 && sexp[1] == :dict
        items = sexp[2]
        pairs = Expr[]
        for i in 1:2:length(items)-1
            k = items[i]
            v = items[i+1]
            push!(pairs, Expr(:call, :(=>), codegen(k), codegen(v)))
        end
        return Expr(:call, :Dict, pairs...)
    end
    
    # List (function call or special form)
    if sexp isa Vector
        isempty(sexp) && return :nothing
        
        head = sexp[1]
        args = sexp[2:end]
        
        # Special forms
        if head == :defn
            # (defn name (args) body) or (defn name (args) "docstring" body)
            name = julia_sym(args[1])
            params_sexp = args[2]
            if params_sexp isa Tuple && params_sexp[1] == :vec
                params = [julia_sym(p) for p in params_sexp[2]]
            else
                error("defn params must be a vector")
            end
            
            # Check for docstring
            if length(args) >= 4 && args[3] isa String
                docstring = args[3]
                body = codegen(args[4])
            else
                docstring = nothing
                body = codegen(args[3])
            end
            
            func_def = Expr(:function, 
                Expr(:call, name, params...),
                body)
            
            if docstring !== nothing
                return Expr(:block,
                    Expr(:macrocall, Symbol("@doc"), LineNumberNode(0), docstring, func_def))
            else
                return func_def
            end
            
        elseif head == :fn
            # (fn (args) body)
            params_sexp = args[1]
            if params_sexp isa Tuple && params_sexp[1] == :vec
                params = [julia_sym(p) for p in params_sexp[2]]
            else
                error("fn params must be a vector")
            end
            body = codegen(args[2])
            return Expr(:->, 
                length(params) == 1 ? params[1] : Expr(:tuple, params...),
                body)
            
        elseif head == :let
            # (let [x 1 y 2] body)
            bindings_sexp = args[1]
            if !(bindings_sexp isa Tuple && bindings_sexp[1] == :vec)
                error("let bindings must be a vector")
            end
            bindings = bindings_sexp[2]
            body = codegen(args[2])
            
            # Build nested let
            result = body
            for i in length(bindings)-1:-2:1
                var = julia_sym(bindings[i])
                val = codegen(bindings[i+1])
                result = Expr(:let, Expr(:(=), var, val), result)
            end
            return result
            
        elseif head == :if
            # (if cond then else)
            cond = codegen(args[1])
            then_branch = codegen(args[2])
            else_branch = length(args) >= 3 ? codegen(args[3]) : :nothing
            return Expr(:if, cond, then_branch, else_branch)
            
        elseif head == :do
            # (do expr1 expr2 ...)
            return Expr(:block, map(codegen, args)...)
            
        elseif head == :quote
            # (quote x) or 'x
            return QuoteNode(args[1])
            
        elseif head == :loop
            # (loop [x init] body with (recur new-x))
            # Compile to while true with break
            bindings_sexp = args[1]
            body = args[2]
            
            if !(bindings_sexp isa Tuple && bindings_sexp[1] == :vec)
                error("loop bindings must be a vector")
            end
            bindings = bindings_sexp[2]
            
            # Create mutable variables
            inits = Expr[]
            vars = Symbol[]
            for i in 1:2:length(bindings)-1
                var = julia_sym(bindings[i])
                val = codegen(bindings[i+1])
                push!(inits, Expr(:(=), var, val))
                push!(vars, var)
            end
            
            # Transform body, replacing (recur ...) with assignments + continue
            transformed_body = transform_recur(codegen(body), vars)
            
            return Expr(:block,
                inits...,
                Expr(:while, true, transformed_body))
            
        elseif head == :for
            # (for [x xs] body)
            bindings_sexp = args[1]
            if !(bindings_sexp isa Tuple && bindings_sexp[1] == :vec)
                error("for bindings must be a vector")
            end
            bindings = bindings_sexp[2]
            var = julia_sym(bindings[1])
            iter = codegen(bindings[2])
            body = codegen(args[2])
            return Expr(:for, Expr(:(=), var, iter), body)
            
        elseif head == :dotimes
            # (dotimes [i n] body)
            bindings_sexp = args[1]
            if !(bindings_sexp isa Tuple && bindings_sexp[1] == :vec)
                error("dotimes bindings must be a vector")
            end
            bindings = bindings_sexp[2]
            var = julia_sym(bindings[1])
            n = codegen(bindings[2])
            body = codegen(args[2])
            return Expr(:for, Expr(:(=), var, Expr(:call, :(:), 1, n)), body)
            
        elseif head == :getfield
            # (getfield obj :field)
            obj = codegen(args[1])
            field = args[2]
            return Expr(:., obj, QuoteNode(field))
            
        elseif head == :setfield!
            # (setfield! obj :field val)
            obj = codegen(args[1])
            field = args[2]
            val = codegen(args[3])
            return Expr(:(=), Expr(:., obj, QuoteNode(field)), val)
            
        # Operators
        elseif head in [:+, :-, :*, :/, :>, :<, :>=, :<=, :(==), :(!=), :and, :or, :mod, :div]
            op = head == :and ? :&& : head == :or ? :|| : head
            if length(args) == 1
                return Expr(:call, op, codegen(args[1]))
            else
                return Expr(:call, op, map(codegen, args)...)
            end
            
        elseif head == :not
            return Expr(:call, :!, codegen(args[1]))
            
        # Default: function call
        else
            func = julia_sym(head)
            return Expr(:call, func, map(codegen, args)...)
        end
    end
    
    error("Cannot generate code for: $sexp ($(typeof(sexp)))")
end

"""
Transform (recur ...) calls into variable assignments + continue.
"""
function transform_recur(expr, vars)
    if expr isa Expr
        if expr.head == :call && expr.args[1] == :recur
            # (recur new-val1 new-val2 ...)
            new_vals = expr.args[2:end]
            assigns = [Expr(:(=), vars[i], new_vals[i]) for i in 1:length(vars)]
            return Expr(:block, assigns..., :(continue))
        else
            return Expr(expr.head, [transform_recur(a, vars) for a in expr.args]...)
        end
    else
        return expr
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Evaluator
# ═══════════════════════════════════════════════════════════════════════════════

function sexp_eval(s::AbstractString, mod::Module=Main)
    sexp = sexp_read(s)
    expr = codegen(sexp)
    Core.eval(mod, expr)
end

"""
    @sx str

Parse and evaluate S-expression string at compile time.

# Example
```julia
@sx "(+ 1 2)"  # → 3
@sx "(defn add [x y] (+ x y))"  # defines add(x, y)
```
"""
macro sx(str)
    sexp = sexp_read(str)
    expr = codegen(sexp)
    esc(expr)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: sexp string macro
# ═══════════════════════════════════════════════════════════════════════════════

"""
    sx"(+ 1 2)"

String macro for S-expressions.
"""
macro sx_str(str)
    sexp = sexp_read(str)
    expr = codegen(sexp)
    esc(expr)
end

export @sx, @sx_str, sexp_read, sexp_eval, sexp_parse

end # module SExp

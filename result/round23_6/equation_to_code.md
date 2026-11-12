# Fixed-PDF P1 mapping

PDF SHA256: 67fcbf68f6d11d45bb94381f80ea44d814cad37035b1daa4fae19f6c1b77c4b4

Equation (5): nonnegative expert weights sum to one.
Equation (6): asset target q = Omega @ w; score convex fusion and subsequent Top2 selection are not this operation.
Equation (7): maximize discounted power utility, gamma=3, discount alpha^(t-1-s). The historical nardari.py solver is a project implementation, not author code.
Equation (9): the PDF displays w*_(t-1)(alpha) inside the sum over s=tau*..t-1, with tau*=tau+tau0. The current fitted vector is used exactly as displayed; it is not replaced by a made-up held-out-return alpha selector. The empirical design reserves 60 expert OOS months for weights and another 60 for initial alpha tuning. Our available frozen expert ledgers start in 2018, so this mathematical weight study never declares adaptive FLEXPOOL active through July 2026. Recovering exact pre-2018 expert strategy ledgers remains incomplete.

Monthly optimization uses only completed earlier months. All monthly outputs are weights only: no shared-cash, netted-order, T+1-constrained account has been implemented. No economic result is attributed to C02-C06. C03 would reuse the exact C01 annual weights, but its asset-space execution is also unfinished. The component-NAV cost model is not claimed equal to the required netted external-order cost model.

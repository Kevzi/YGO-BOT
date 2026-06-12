## Deferred from: code review of 1-1-base-de-donnees-et-historisation (2026-06-12)

- Validation JSON : Conservé en Dict[str, Any] pour l'instant. La structure sera décidée avec YGO Omega/Neos (Story 1.3) et ygoenv (Story 2.1).
- Session DB applicative : La création de la session de base de données (get_db) est reportée à la création du serveur API (Story 1.3) pour s'intégrer avec l'injection de dépendances de FastAPI.

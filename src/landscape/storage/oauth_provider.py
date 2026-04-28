"""LandscapeOAuthProvider — OAuth 2.1 authorization server backed by SQLite."""
from __future__ import annotations

import secrets
import time
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from mcp.server.auth.provider import (
    AccessToken as SDKAccessToken,
)
from mcp.server.auth.provider import (
    AuthorizationCode as SDKAuthorizationCode,
)
from mcp.server.auth.provider import (
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
)
from mcp.server.auth.provider import (
    RefreshToken as SDKRefreshToken,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import AnyUrl

from landscape.storage import auth_store


class LandscapeAuthorizationCode(SDKAuthorizationCode):
    pass


class LandscapeAccessToken(SDKAccessToken):
    token_id: str
    client_name: str


class LandscapeRefreshToken(SDKRefreshToken):
    token_id: str


def _new_token() -> str:
    return secrets.token_urlsafe(32)


def _is_public_client(client: OAuthClientInformationFull) -> bool:
    return client.token_endpoint_auth_method == "none" or not client.client_secret


class LandscapeOAuthProvider(
    OAuthAuthorizationServerProvider[
        LandscapeAuthorizationCode,
        LandscapeRefreshToken,
        LandscapeAccessToken,
    ]
):
    """OAuth 2.1 provider — single-user, local-first, auto-approving."""

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return await auth_store.get_oauth_client(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        await auth_store.store_oauth_client(client_info)

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        """Auto-approve: mint an auth code and redirect immediately."""
        code = _new_token()
        expires_at = time.time() + 600
        scopes = params.scopes or (client.scope.split() if client.scope else ["agent"])
        rupe = getattr(params, "redirect_uri_provided_explicitly", True)
        # rupe may be None if the SDK field exists but wasn't set
        if rupe is None:
            rupe = True

        await auth_store.store_authorization_code(
            code=code,
            client_id=client.client_id,
            redirect_uri=str(params.redirect_uri),
            redirect_uri_provided_explicitly=bool(rupe),
            scopes=scopes,
            code_challenge=params.code_challenge,
            expires_at=expires_at,
        )

        parsed = urlparse(str(params.redirect_uri))
        query = parse_qs(parsed.query)
        query["code"] = [code]
        state = getattr(params, "state", None)
        if state:
            query["state"] = [state]
        new_query = urlencode({k: v[0] for k, v in query.items()})
        return urlunparse(parsed._replace(query=new_query))

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> LandscapeAuthorizationCode | None:
        row = await auth_store.load_authorization_code_record(authorization_code)
        if row is None:
            return None
        return LandscapeAuthorizationCode(
            code=row["code"],
            client_id=row["client_id"],
            redirect_uri=AnyUrl(row["redirect_uri"]),
            redirect_uri_provided_explicitly=row["redirect_uri_provided_explicitly"],
            scopes=row["scopes"],
            code_challenge=row["code_challenge"],
            expires_at=row["expires_at"],
            resource=None,
        )

    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: LandscapeAuthorizationCode,
    ) -> OAuthToken:
        """PKCE is verified by the SDK's TokenHandler before this is called."""
        marked = await auth_store.mark_code_used(authorization_code.code)
        if not marked:
            raise ValueError("authorization code already used")

        token_id = _new_token()
        access_token = _new_token()
        refresh_token = _new_token()
        client_name = client.client_name or client.client_id

        await auth_store.store_oauth_token(
            token_id=token_id,
            client_id=client.client_id,
            client_name=client_name,
            access_token=access_token,
            refresh_token=refresh_token,
            scopes=authorization_code.scopes,
            expires_at=None,
        )
        return OAuthToken(
            access_token=access_token,
            token_type="Bearer",
            refresh_token=refresh_token,
            scope=" ".join(authorization_code.scopes),
        )

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> LandscapeRefreshToken | None:
        row = await auth_store.load_oauth_token_by_refresh(refresh_token)
        if row is None:
            if not _is_public_client(client):
                return None
            stale_row = await auth_store.load_oauth_token_record_by_refresh(refresh_token)
            if stale_row is None or stale_row["client_id"] != client.client_id:
                return None
            if stale_row["revoked_at"] is None:
                return None
            row = await auth_store.load_latest_live_oauth_token_by_client_id(
                client.client_id
            )
            if row is None:
                return None
        return LandscapeRefreshToken(
            token=row["refresh_token"],   # SDK field name is 'token'
            token_id=row["token_id"],
            client_id=row["client_id"],
            scopes=row["scopes"],
            expires_at=row["expires_at"],
        )

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: LandscapeRefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        """Refresh access tokens, preserving refresh tokens for public clients."""
        effective_scopes = scopes if scopes else refresh_token.scopes
        access_token = _new_token()
        client_name = client.client_name or client.client_id

        if _is_public_client(client):
            replaced = await auth_store.replace_access_token(
                token_id=refresh_token.token_id,
                client_name=client_name,
                access_token=access_token,
                scopes=effective_scopes,
                expires_at=None,
            )
            if not replaced:
                raise ValueError("refresh token no longer live")
            new_refresh_token = refresh_token.token
        else:
            await auth_store.revoke_oauth_token_by_id(refresh_token.token_id)

            token_id = _new_token()
            new_refresh_token = _new_token()
            await auth_store.store_oauth_token(
                token_id=token_id,
                client_id=client.client_id,
                client_name=client_name,
                access_token=access_token,
                refresh_token=new_refresh_token,
                scopes=effective_scopes,
                expires_at=None,
            )
        return OAuthToken(
            access_token=access_token,
            token_type="Bearer",
            refresh_token=new_refresh_token,
            scope=" ".join(effective_scopes),
        )

    async def load_access_token(self, token: str) -> LandscapeAccessToken | None:
        row = await auth_store.load_oauth_token_by_access(token)
        if row is None:
            return None
        # SDK AccessToken uses 'token' field name (not 'access_token')
        expires_raw = row["expires_at"]
        expires_at = int(expires_raw) if expires_raw is not None else None
        return LandscapeAccessToken(
            token=row["access_token"],
            client_id=row["client_id"],
            scopes=row["scopes"],
            expires_at=expires_at,
            token_id=row["token_id"],
            client_name=row["client_name"],
        )

    async def revoke_token(
        self, token: LandscapeAccessToken | LandscapeRefreshToken
    ) -> None:
        await auth_store.revoke_oauth_token_by_id(token.token_id)

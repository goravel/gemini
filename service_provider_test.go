package gemini

import (
	"testing"

	contractsai "github.com/goravel/framework/contracts/ai"
	contractsfoundation "github.com/goravel/framework/contracts/foundation"
	mocksconfig "github.com/goravel/framework/mocks/config"
	mocksfoundation "github.com/goravel/framework/mocks/foundation"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

func TestServiceProviderRegisterReturnsErrorForMissingProviderParameter(t *testing.T) {
	app, _ := mockServiceProviderApp(t)

	callback := registerCallback(t, app)
	instance, err := callback(app, map[string]any{})

	require.Error(t, err)
	assert.Nil(t, instance)
	assert.Equal(t, "missing gemini provider parameter", err.Error())
}

func TestServiceProviderRegisterBuildsProviderFromParameter(t *testing.T) {
	app, config := mockServiceProviderApp(t)
	config.EXPECT().UnmarshalKey("ai.providers.gemini", new(contractsai.ProviderConfig)).RunAndReturn(func(key string, rawVal any) error {
		cfg := rawVal.(*contractsai.ProviderConfig)
		assert.Equal(t, "ai.providers.gemini", key)
		cfg.Key = "test-key"
		return nil
	}).Once()

	callback := registerCallback(t, app)
	instance, err := callback(app, map[string]any{"provider": "gemini"})

	require.NoError(t, err)
	_, ok := instance.(contractsai.Provider)
	assert.True(t, ok)
}

func mockServiceProviderApp(t *testing.T) (*mocksfoundation.Application, *mocksconfig.Config) {
	t.Helper()

	app := mocksfoundation.NewApplication(t)
	config := mocksconfig.NewConfig(t)
	app.EXPECT().MakeConfig().Return(config).Once()

	return app, config
}

func registerCallback(t *testing.T, app *mocksfoundation.Application) func(contractsfoundation.Application, map[string]any) (any, error) {
	t.Helper()

	provider := &ServiceProvider{}
	var callback func(contractsfoundation.Application, map[string]any) (any, error)
	app.EXPECT().BindWith(Binding, mock.MatchedBy(func(bindingCallback func(contractsfoundation.Application, map[string]any) (any, error)) bool {
		return bindingCallback != nil
	})).Run(func(_ any, bindingCallback func(contractsfoundation.Application, map[string]any) (any, error)) {
		callback = bindingCallback
	}).Once()

	provider.Register(app)
	require.NotNil(t, callback)

	return callback
}
